from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
import torch
from diffusers import KandinskyV22Pipeline, KandinskyV22PriorPipeline
from PIL import Image
import io
import os
import json
import requests
import base64
import uuid
from dotenv import load_dotenv
import logging
from typing import Optional, List
import boto3
from botocore.exceptions import NoCredentialsError
import gradio as gr
import asyncio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Log to console
        logging.FileHandler("app.log")  # Log to file as well
    ]
)
logger = logging.getLogger("esmart-ai-service")

# Set up device - using CUDA for Hugging Face Spaces
if torch.cuda.is_available():
    device = "cuda"
    logger.info("Using CUDA for GPU acceleration")
    # Configure to optimize VRAM
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
else:
    device = "cpu"
    logger.info("Using CPU for computation")

# Set default device
torch.set_default_device(device)

# Load environment variables
load_dotenv()

# Configure S3 or similar for cloud storage
USE_CLOUD_STORAGE = os.getenv("USE_CLOUD_STORAGE", "False").lower() == "true"
S3_BUCKET = os.getenv("S3_BUCKET", "")
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY", "")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY", "")
S3_REGION = os.getenv("S3_REGION", "us-east-1")

# Configure API
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", "512"))  # Limit maximum size
DEFAULT_STEPS = int(os.getenv("DEFAULT_STEPS", "20"))  # Reduce default steps to save resources

# Initialize FastAPI
app = FastAPI(title="Esmart AI Image Generator API")

# Set up CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
PRIOR_MODEL_ID = "kandinsky-community/kandinsky-2-2-prior"
MODEL_ID = "kandinsky-community/kandinsky-2-2-decoder"
OUTPUT_DIR = "outputs"

# Ensure outputs directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Lazy loading models to optimize memory
prior_pipe = None
pipe = None

# Initialize S3 client if needed
s3_client = None
if USE_CLOUD_STORAGE and S3_ACCESS_KEY and S3_SECRET_KEY and S3_BUCKET:
    try:
        s3_client = boto3.client(
            's3',
            aws_access_key_id=S3_ACCESS_KEY,
            aws_secret_access_key=S3_SECRET_KEY,
            region_name=S3_REGION
        )
        logger.info("S3 client initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing S3 client: {str(e)}")

def free_memory():
    """Free memory when needed"""
    global prior_pipe, pipe
    
    if device == "cuda":
        # Clear CUDA cache
        torch.cuda.empty_cache()
        
    # Free models if loaded
    if prior_pipe is not None:
        del prior_pipe
        prior_pipe = None
        
    if pipe is not None:
        del pipe
        pipe = None
        
    logger.info("Memory freed successfully")

def load_prior_model():
    global prior_pipe
    if prior_pipe is None:
        try:
            logger.info("Loading Kandinsky 2.2 Prior model...")
            # Use torch.float16 to reduce memory requirements
            prior_pipe = KandinskyV22PriorPipeline.from_pretrained(
                PRIOR_MODEL_ID, 
                torch_dtype=torch.float16,
                use_safetensors=True
            )
            prior_pipe = prior_pipe.to(device)
            logger.info(f"Prior model loaded successfully on {device}")
        except Exception as e:
            logger.error(f"Error loading prior model: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")
    return prior_pipe

def load_model():
    global pipe
    if pipe is None:
        try:
            logger.info("Loading Kandinsky 2.2 Decoder model...")
            # Use torch.float16 and attention slicing to reduce memory
            pipe = KandinskyV22Pipeline.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.float16,
                use_safetensors=True
            )
            # Add attention slicing to reduce memory footprint
            if hasattr(pipe, "enable_attention_slicing"):
                pipe.enable_attention_slicing()
            pipe = pipe.to(device)
            logger.info(f"Decoder model loaded successfully on {device}")
        except Exception as e:
            logger.error(f"Error loading decoder model: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")
    return pipe

def save_image(image, prompt):
    """Save image to outputs directory or cloud storage"""
    filename = f"{uuid.uuid4()}.png"
    filepath = os.path.join(OUTPUT_DIR, filename)
    
    # Save image locally
    image.save(filepath)
    logger.info(f"Image saved locally to {filepath}")
    
    # If cloud storage is configured, also save there
    image_url = f"/images/{filename}"  # Default URL
    
    if USE_CLOUD_STORAGE and s3_client:
        try:
            # Convert image to bytes
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)
            
            # Upload to S3
            s3_client.upload_fileobj(
                img_byte_arr, 
                S3_BUCKET, 
                f"images/{filename}",
                ExtraArgs={'ContentType': 'image/png'}
            )
            
            # URL of image on S3
            image_url = f"https://{S3_BUCKET}.s3.amazonaws.com/images/{filename}"
            logger.info(f"Image uploaded to S3: {image_url}")
        except NoCredentialsError:
            logger.error("S3 credentials not available")
        except Exception as e:
            logger.error(f"Error uploading to S3: {str(e)}")
    
    return filepath, filename, image_url

async def generate_improved_prompt(original_prompt):
    """Use DeepSeek to improve prompt"""
    if not OPENROUTER_API_KEY:
        logger.warning("OpenRouter API key not found, using original prompt")
        return original_prompt

    try:
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "deepseek-ai/deepseek-llm-67b-chat",
            "messages": [
                {"role": "system", "content": "You are an AI image prompt engineer. Improve the given text to create a detailed, vivid image prompt for an AI image generator. Add more details, artistic style, lighting, mood, and visual elements to make it more specific and visually interesting. Respond only with the improved prompt, no explanations."},
                {"role": "user", "content": original_prompt}
            ],
            "max_tokens": 500
        }
        
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=data
        )
        
        if response.status_code == 200:
            improved_prompt = response.json()["choices"][0]["message"]["content"]
            logger.info(f"Original prompt: {original_prompt}")
            logger.info(f"Improved prompt: {improved_prompt}")
            return improved_prompt
        else:
            logger.error(f"Error from OpenRouter API: {response.text}")
            return original_prompt
    except Exception as e:
        logger.error(f"Error generating improved prompt: {str(e)}")
        return original_prompt

# Sync version of improve prompt for Gradio
def generate_improved_prompt_sync(original_prompt):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result = loop.run_until_complete(generate_improved_prompt(original_prompt))
    loop.close()
    return result

@app.get("/api")
async def api_root():
    return {"message": "Esmart AI Image Generator API is running"}

@app.get("/api/health")
async def health_check():
    """Health check endpoint for HF Spaces monitoring"""
    return {"status": "healthy", "version": "1.0.0"}

@app.get("/api/status")
async def check_status():
    """Check status of API and models"""
    cuda_available = torch.cuda.is_available()
    cuda_device = torch.cuda.get_device_name(0) if cuda_available else "N/A"
    cuda_memory = None
    
    if cuda_available:
        try:
            cuda_memory = {
                "total": torch.cuda.get_device_properties(0).total_memory / (1024**3),  # GB
                "allocated": torch.cuda.memory_allocated(0) / (1024**3),  # GB
                "cached": torch.cuda.memory_reserved(0) / (1024**3)  # GB
            }
        except:
            cuda_memory = "Error getting memory info"
    
    return {
        "status": "running",
        "cuda_available": cuda_available,
        "device": device,
        "cuda_device": cuda_device,
        "cuda_memory": cuda_memory,
        "models": {
            "prior": prior_pipe is not None,
            "decoder": pipe is not None
        },
        "cloud_storage": USE_CLOUD_STORAGE and s3_client is not None
    }

@app.post("/api/generate-image")
async def generate_image(
    prompt: str = Form(...),
    negative_prompt: Optional[str] = Form("low quality, blurry"),
    enhance_prompt: Optional[bool] = Form(False),
    width: Optional[int] = Form(512),
    height: Optional[int] = Form(512),
    num_inference_steps: Optional[int] = Form(DEFAULT_STEPS),
    guidance_scale: Optional[float] = Form(7.5),
    style: Optional[str] = Form("Simple"),
    background_tasks: BackgroundTasks = None
):
    """
    Generate image from text prompt using Kandinsky 2.2
    - prompt: Description of the image to generate
    - negative_prompt: What should not appear in the image
    - width, height: Image dimensions (pixels)
    - num_inference_steps: Number of inference steps (higher = better quality but slower)
    - guidance_scale: How closely to follow prompt (higher = follows prompt more closely)
    - style: Art style
    """
    try:
        # Limit image size to save resources
        width = min(width, MAX_IMAGE_SIZE)
        height = min(height, MAX_IMAGE_SIZE)
        
        # Improve prompt if requested
        if enhance_prompt:
            # Add style to original prompt
            styled_prompt = f"{prompt}, {style} style"
            final_prompt = await generate_improved_prompt(styled_prompt)
        else:
            # If not improving, still add style to prompt
            final_prompt = f"{prompt}, {style} style"
        
        # Load models if not loaded
        prior = load_prior_model()
        decoder = load_model()
        
        # Generate image embeddings with prior model
        logger.info(f"Generating image embedding for prompt: {final_prompt}")
        image_embeds, negative_image_embeds = prior(
            prompt=final_prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps
        ).to_tuple()
        
        # Generate image from embeddings with decoder model
        logger.info("Generating image from embeddings...")
        image = decoder(
            image_embeds=image_embeds,
            negative_image_embeds=negative_image_embeds,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps
        ).images[0]
        
        # Save image to directory and return path
        filepath, filename, image_url = save_image(image, final_prompt)
        
        # Convert image to base64 to return
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        # Free memory if there are background tasks
        if background_tasks:
            background_tasks.add_task(free_memory)
        
        return JSONResponse({
            "success": True,
            "prompt": final_prompt,
            "filename": filename,
            "image_url": image_url,
            "image_data": img_str
        })
        
    except Exception as e:
        logger.error(f"Error generating image: {str(e)}")
        # Free memory in case of error
        free_memory()
        raise HTTPException(status_code=500, detail=f"Error generating image: {str(e)}")

@app.post("/api/topic-to-image")
async def topic_to_image(
    topic: str = Form(...),
    style: Optional[str] = Form("Simple"),
    width: Optional[int] = Form(512),
    height: Optional[int] = Form(512),
    num_inference_steps: Optional[int] = Form(DEFAULT_STEPS),
    background_tasks: BackgroundTasks = None
):
    """
    Generate image from a topic by creating a prompt from the topic first
    """
    try:
        # Limit image size
        width = min(width, MAX_IMAGE_SIZE)
        height = min(height, MAX_IMAGE_SIZE)
        
        # Create a detailed prompt from topic using DeepSeek
        prompt_prefix = f"Create a visually striking image of {topic} in {style} style."
        improved_prompt = await generate_improved_prompt(prompt_prefix)
        
        # Load models if not loaded
        prior = load_prior_model()
        decoder = load_model()
        
        # Generate image embeddings with prior model
        logger.info(f"Generating image embedding for topic: {topic}")
        image_embeds, negative_image_embeds = prior(
            prompt=improved_prompt,
            negative_prompt="low quality, blurry",
            guidance_scale=7.5,
            num_inference_steps=num_inference_steps
        ).to_tuple()
        
        # Generate image from embeddings with decoder model
        logger.info("Generating image from embeddings...")
        image = decoder(
            image_embeds=image_embeds,
            negative_image_embeds=negative_image_embeds,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps
        ).images[0]
        
        # Save image to directory and return path
        filepath, filename, image_url = save_image(image, improved_prompt)
        
        # Convert image to base64 to return
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        # Free memory if there are background tasks
        if background_tasks:
            background_tasks.add_task(free_memory)
        
        return JSONResponse({
            "success": True,
            "topic": topic,
            "prompt": improved_prompt,
            "filename": filename,
            "image_url": image_url,
            "image_data": img_str
        })
        
    except Exception as e:
        logger.error(f"Error generating image from topic: {str(e)}")
        # Free memory in case of error
        free_memory()
        raise HTTPException(status_code=500, detail=f"Error generating image from topic: {str(e)}")

@app.get("/api/images/{filename}")
async def get_image(filename: str):
    """Return a generated image by filename"""
    filepath = os.path.join(OUTPUT_DIR, filename)
    if not os.path.isfile(filepath):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(filepath)

@app.delete("/api/images/{filename}")
async def delete_image(filename: str):
    """Delete a generated image by filename"""
    filepath = os.path.join(OUTPUT_DIR, filename)
    if not os.path.isfile(filepath):
        return JSONResponse({
            "success": False,
            "message": "Image not found",
        }, status_code=404)
    
    try:
        os.remove(filepath)
        logger.info(f"Deleted image: {filename}")
        
        # If S3 exists, also delete image there
        if USE_CLOUD_STORAGE and s3_client:
            try:
                s3_client.delete_object(Bucket=S3_BUCKET, Key=f"images/{filename}")
                logger.info(f"Deleted image from S3: {filename}")
            except Exception as e:
                logger.error(f"Error deleting image from S3: {str(e)}")
        
        return JSONResponse({
            "success": True,
            "message": f"Successfully deleted image: {filename}",
        })
    except Exception as e:
        logger.error(f"Error deleting image {filename}: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error deleting image: {str(e)}"
        )

# Wrapper functions for Gradio
def generate_image_gradio(prompt, negative_prompt, width, height, style, steps, enhance):
    try:
        # Validate inputs
        if not prompt:
            return None, "Prompt cannot be empty"
        
        # Limit image dimensions
        width = min(int(width), MAX_IMAGE_SIZE)
        height = min(int(height), MAX_IMAGE_SIZE)
        
        # Improve prompt if requested
        if enhance:
            styled_prompt = f"{prompt}, {style} style"
            final_prompt = generate_improved_prompt_sync(styled_prompt)
        else:
            final_prompt = f"{prompt}, {style} style"
        
        # Load models
        prior = load_prior_model()
        decoder = load_model()
        
        # Generate image embeddings
        image_embeds, negative_image_embeds = prior(
            prompt=final_prompt,
            negative_prompt=negative_prompt,
            guidance_scale=7.5,
            num_inference_steps=int(steps)
        ).to_tuple()
        
        # Generate image
        image = decoder(
            image_embeds=image_embeds,
            negative_image_embeds=negative_image_embeds,
            height=height,
            width=width,
            num_inference_steps=int(steps)
        ).images[0]
        
        # Save image
        filepath, filename, _ = save_image(image, final_prompt)
        
        # Free memory in background
        free_memory()
        
        return image, final_prompt
    
    except Exception as e:
        error_msg = f"Error generating image: {str(e)}"
        logger.error(error_msg)
        free_memory()
        return None, error_msg

def topic_to_image_gradio(topic, style, width, height, steps):
    try:
        if not topic:
            return None, "Topic cannot be empty"
        
        # Limit image dimensions
        width = min(int(width), MAX_IMAGE_SIZE)
        height = min(int(height), MAX_IMAGE_SIZE)
        
        # Generate prompt from topic
        prompt_prefix = f"Create a visually striking image of {topic} in {style} style."
        improved_prompt = generate_improved_prompt_sync(prompt_prefix)
        
        # Load models
        prior = load_prior_model()
        decoder = load_model()
        
        # Generate image embeddings
        image_embeds, negative_image_embeds = prior(
            prompt=improved_prompt,
            negative_prompt="low quality, blurry",
            guidance_scale=7.5,
            num_inference_steps=int(steps)
        ).to_tuple()
        
        # Generate image
        image = decoder(
            image_embeds=image_embeds,
            negative_image_embeds=negative_image_embeds,
            height=height,
            width=width,
            num_inference_steps=int(steps)
        ).images[0]
        
        # Save image
        filepath, filename, _ = save_image(image, improved_prompt)
        
        # Free memory in background
        free_memory()
        
        return image, improved_prompt
    
    except Exception as e:
        error_msg = f"Error generating image from topic: {str(e)}"
        logger.error(error_msg)
        free_memory()
        return None, error_msg

# Create Gradio interface
def create_gradio_interface():
    with gr.Blocks(title="Esmart AI Image Generator") as demo:
        gr.Markdown("# Esmart AI Image Generator")
        gr.Markdown("Generate beautiful images with Kandinsky 2.2 AI model")
        
        with gr.Tab("Text to Image"):
            with gr.Row():
                with gr.Column():
                    prompt = gr.Textbox(label="Prompt", placeholder="Describe the image you want to generate")
                    neg_prompt = gr.Textbox(label="Negative Prompt", value="low quality, blurry")
                    
                    with gr.Row():
                        width = gr.Slider(minimum=256, maximum=768, value=512, step=64, label="Width")
                        height = gr.Slider(minimum=256, maximum=768, value=512, step=64, label="Height")
                    
                    style = gr.Dropdown(["Simple", "Photographic", "Digital Art", "Watercolor", "Oil Painting", "Cinematic"], label="Style", value="Simple")
                    steps = gr.Slider(minimum=10, maximum=50, value=20, step=1, label="Inference Steps")
                    enhance = gr.Checkbox(label="Enhance Prompt", value=True)
                    generate_btn = gr.Button("Generate Image")
                
                with gr.Column():
                    output_image = gr.Image(type="pil", label="Generated Image")
                    output_prompt = gr.Textbox(label="Final Prompt")
            
            generate_btn.click(
                fn=generate_image_gradio,
                inputs=[prompt, neg_prompt, width, height, style, steps, enhance],
                outputs=[output_image, output_prompt]
            )
        
        with gr.Tab("Topic to Image"):
            with gr.Row():
                with gr.Column():
                    topic = gr.Textbox(label="Topic", placeholder="Enter a topic or concept")
                    topic_style = gr.Dropdown(["Simple", "Photographic", "Digital Art", "Watercolor", "Oil Painting", "Cinematic"], label="Style", value="Simple")
                    
                    with gr.Row():
                        topic_width = gr.Slider(minimum=256, maximum=768, value=512, step=64, label="Width")
                        topic_height = gr.Slider(minimum=256, maximum=768, value=512, step=64, label="Height")
                    
                    topic_steps = gr.Slider(minimum=10, maximum=50, value=20, step=1, label="Inference Steps")
                    topic_btn = gr.Button("Generate Image from Topic")
                
                with gr.Column():
                    topic_output_image = gr.Image(type="pil", label="Generated Image")
                    topic_output_prompt = gr.Textbox(label="Generated Prompt")
            
            topic_btn.click(
                fn=topic_to_image_gradio,
                inputs=[topic, topic_style, topic_width, topic_height, topic_steps],
                outputs=[topic_output_image, topic_output_prompt]
            )
            
        with gr.Tab("API Information"):
            gr.Markdown("""
            ## API Endpoints
            
            This service also provides REST API endpoints:
            
            - `GET /api/status`: Check the status of the service
            - `POST /api/generate-image`: Generate an image from a text prompt
            - `POST /api/topic-to-image`: Generate an image from a topic
            - `GET /api/images/{filename}`: Get a generated image by filename
            
            For API documentation, visit: [API Docs](/docs)
            """)
            
        gr.Markdown("Created by Esmart AI")
    
    return demo

# Mount Gradio interface to FastAPI app
demo = create_gradio_interface()
app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=True) 