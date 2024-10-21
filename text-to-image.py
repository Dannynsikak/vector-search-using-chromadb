import torch
import chromadb
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import gradio as gr
import time
from sklearn.metrics.pairwise import cosine_similarity

# TODO#2: Setup ChromaDB for Image and Text Collections
client = chromadb.Client()
image_collection = client.create_collection("image_collection")
text_collection = client.create_collection("text_collection")

# TODO#3: Load CLIP model and processor for generating image and text embeddings
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# TODO#4: Load and preprocess images
image_paths = [
    "img/—Pngtree—brave firefighter clip art woman_13810240.png", 
    "img/—Pngtree—movement basketball logo_17325217.png",
    "img/cholera.jpg", 
    "img/malaria.webp",
    "img/ovariansyndrome.png",
    "img/Tuberculosis.jpg",
    "img/waistpain.jpeg",
    "img/asthma.jpeg",
    "img/Epilepsy-monitoring-evaluation.webp"
]

# Preprocess images and generate embeddings
images = [Image.open(image_path).convert("RGBA") for image_path in image_paths]
image_inputs = clip_processor(images=images, return_tensors="pt", padding=True)

start_ingestion_time = time.time()
with torch.no_grad():
    image_embeddings = clip_model.get_image_features(**image_inputs).numpy()
image_embeddings = [embedding.tolist() for embedding in image_embeddings]
image_collection.add(
    embeddings=image_embeddings,
    metadatas=[{"image": image_path} for image_path in image_paths],
    ids=[str(i) for i in range(len(image_paths))]
)

medical_texts = [
    "A 45-year-old male diagnosed with Type 2 Diabetes, presenting with symptoms of fatigue and increased thirst.",
    "A patient suffering from Alzheimer's disease, exhibiting memory loss and difficulty performing familiar tasks.",
    "A case of Acute Lymphoblastic Leukemia in a child, characterized by fever, fatigue, and frequent infections.",
    "A 60-year-old female with Chronic Obstructive Pulmonary Disease (COPD), experiencing shortness of breath and persistent cough.",
    "A newly diagnosed case of Rheumatoid Arthritis with joint pain, swelling, and morning stiffness.",
    "A patient with Myocardial Infarction (Heart Attack) presenting chest pain, shortness of breath, and sweating.",
    "A 30-year-old woman diagnosed with Polycystic Ovary Syndrome (PCOS) showing irregular periods and weight gain.",
    "A 12-year-old child diagnosed with Asthma, experiencing wheezing, coughing, and difficulty breathing during physical activity.",
    "A patient presenting symptoms of Epilepsy, characterized by recurrent seizures and loss of consciousness.",
    "A case of Tuberculosis (TB) with a persistent cough, weight loss, and night sweats."
]

# Preprocess text and generate embeddings
text_inputs = clip_processor(text=medical_texts, return_tensors="pt", padding=True)

with torch.no_grad():
    text_embeddings = clip_model.get_text_features(**text_inputs).numpy()

text_embeddings = [embedding.tolist() for embedding in text_embeddings]
text_collection.add(
    embeddings=text_embeddings,
    metadatas=[{"text": text} for text in medical_texts],
    ids=[str(i) for i in range(len(medical_texts))]
)

# TODO#6: Search Function for Images and Texts
def search(query, search_type):
    start_time = time.time()
    try:
        if search_type == "image":
            # Process the text query into image embeddings
            inputs = clip_processor(text=query, return_tensors="pt", padding=True)
            with torch.no_grad():
                query_embedding = clip_model.get_text_features(**inputs).numpy().tolist()

            # Query the image collection
            results = image_collection.query(query_embeddings=query_embedding, n_results=1)

            if results['ids']:
                result_image_path = results["metadatas"][0][0]["image"]
                result_image = Image.open(result_image_path)
                accuracy_score = cosine_similarity([image_embeddings[int(results["ids"][0][0])]], query_embedding)[0][0]
                query_time = time.time() - start_time
                return result_image, f"Accuracy: {accuracy_score:.4f}", f"Time: {query_time:.4f} seconds"
            else:
                return None, "No images found for your query.", f"Time: {time.time() - start_time:.4f} seconds"

        elif search_type == "text":
            # Process the text query into text embeddings
            inputs = clip_processor(text=query, return_tensors="pt", padding=True)
            with torch.no_grad():
                query_embedding = clip_model.get_text_features(**inputs).numpy().tolist()

            results = text_collection.query(query_embeddings=[query_embedding], n_results=1)

            if results['ids']:
                result_text = results["metadatas"][0][0]["text"]
                accuracy_score = cosine_similarity([text_embeddings[int(results["ids"][0][0])]], query_embedding)[0][0]
                query_time = time.time() - start_time
                return None, result_text, f"Accuracy: {accuracy_score:.4f}\nTime: {query_time:.4f} seconds"
            else:
                return None, "No text documents found for your query.", f"Time: {time.time() - start_time:.4f} seconds"

    except Exception as e:
        return None, f"An error occurred while searching: {str(e)}", f"Time: {time.time() - start_time:.4f} seconds"

# TODO#7: Define Gradio Interface
with gr.Blocks() as gr_interface:
    gr.Markdown("# Medical Document and Image Search using ChromaDB")

    with gr.Row():
        with gr.Column():
            custom_query = gr.Textbox(label="Enter query (text or image search)", placeholder="Enter symptoms, diagnosis or a query")
            search_type = gr.Radio(choices=["image", "text"], label="Search Type", value="text")
            submit_button = gr.Button("Search")

        with gr.Column():
            output_image = gr.Image(label="Retrieved Image (if image search)", interactive=False)
            output_text = gr.Textbox(label="Retrieved Document (if text search)", lines=5)
            performance_metrics = gr.Textbox(label="Performance Metrics", lines=3)

    # Set up Gradio function for the submit button
    submit_button.click(
        fn=search, 
        inputs=[custom_query, search_type], 
        outputs=[output_image, output_text, performance_metrics]
    )

# TODO#8: Launch Gradio Interface
gr_interface.launch()
