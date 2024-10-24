import torch
import chromadb
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import gradio as gr
import time
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

# Setup ChromaDB for Image and Text Collections
client = chromadb.Client()
image_collection = client.create_collection("image_collection")
text_collection = client.create_collection("text_collection")

# Load CLIP model and processor for generating image and text embeddings
try:
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    print("CLIP model and processor loaded successfully.")
except Exception as e:
    print(f"Error loading CLIP model: {e}")

# Function to preprocess and generate embeddings for images
def process_images(image_paths):
    images = []
    for image_path in image_paths:
        try:
            image = Image.open(image_path).convert("RGB")
            images.append(image)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")

    image_inputs = clip_processor(images=images, return_tensors="pt", padding=True)
    
    with torch.no_grad():
        image_embeddings = clip_model.get_image_features(**image_inputs).numpy()
    
    # Normalize embeddings for better similarity accuracy
    image_embeddings = normalize(image_embeddings)
    return image_embeddings.tolist()

# Function to preprocess and generate embeddings for text
def process_texts(texts):
    text_inputs = clip_processor(text=texts, return_tensors="pt", padding=True, truncation=True, max_length=77)
    
    with torch.no_grad():
        text_embeddings = clip_model.get_text_features(**text_inputs).numpy()

    text_embeddings = normalize(text_embeddings)
    return text_embeddings.tolist()

# Function to calculate similarity (accuracy score)
def calculate_similarity(image_embedding, query_embedding):
    similarity = cosine_similarity([image_embedding], [query_embedding])[0][0]
    return similarity

# Function to add embeddings to ChromaDB collections
def add_to_collection(collection, embeddings, metadatas, ids):
    start_time = time.time()  # Start time for adding to collection
    collection.add(embeddings=embeddings, metadatas=metadatas, ids=ids)
    end_time = time.time()  # End time for adding to collection
    time_taken = f"Time taken to add to collection: {end_time - start_time:.4f} seconds"
    return time_taken

# Load and preprocess images
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
start_ingestion_time = time.time()
image_embeddings = process_images(image_paths)

# Add image embeddings to the ChromaDB collection
ingestion_time_taken = add_to_collection(
    image_collection, 
    embeddings=image_embeddings, 
    metadatas=[{"image": image_path} for image_path in image_paths], 
    ids=[str(i) for i in range(len(image_paths))]
)

ingestion_time = time.time() - start_ingestion_time
print(f"Image Data ingestion time: {ingestion_time:.4f} seconds")

# Function to convert medical case dictionary into a descriptive string
def generate_medical_case(disease):
    description = (
        f"Disease Name: {disease['Disease Name']}\n"
        f"Alternate Names: {disease['Alternate Names']}\n"
        f"Overview: {disease['Overview']}\n"
        f"Causes: {disease['Causes']}\n"
        f"Symptoms: {disease['Symptoms']}\n"
        f"Diagnosis: {disease['Diagnosis']}\n"
        f"Treatment: {disease['Treatment']}\n"
        f"Prevention: {disease['Prevention']}\n"
        f"Complications: {disease['Complications']}\n"
        f"Prognosis: {disease['Prognosis']}\n"
    )
    return description

# Define medical texts for embedding and preprocess
medical_texts = [
   {
    'Disease Name': 'Chronic Obstructive Pulmonary Disease (COPD)',
    'Alternate Names': 'Chronic Bronchitis, Emphysema',
    'Overview': 'COPD is a group of progressive lung diseases that cause breathing difficulties.',
    'Causes': 'Long-term exposure to irritants like tobacco smoke, air pollution, genetic factors',
    'Symptoms': 'Shortness of breath, Wheezing, Chronic cough',
    'Diagnosis': 'Spirometry, Chest X-ray, Blood gas analysis',
    'Treatment': 'Bronchodilators, Steroids, Oxygen therapy, Surgery in severe cases',
    'Prevention': 'Avoid smoking, Reduce exposure to lung irritants',
    'Complications': 'Respiratory infections, Heart problems, Lung cancer',
    'Prognosis': 'Progressive disease with no cure, but manageable with proper treatment.'
},
{
    'Disease Name': 'Asthma',
    'Alternate Names': 'Bronchial Asthma',
    'Overview': 'Asthma is a condition in which airways narrow and swell, causing breathing difficulties.',
    'Causes': 'Allergies, Respiratory infections, Exercise, Environmental factors',
    'Symptoms': 'Wheezing, Shortness of breath, Chest tightness, Coughing',
    'Diagnosis': 'Spirometry, Peak flow test, Allergy testing',
    'Treatment': 'Inhaled corticosteroids, Bronchodilators, Avoiding triggers',
    'Prevention': 'Avoiding known triggers, Regular exercise, Allergy management',
    'Complications': 'Frequent hospitalizations, Respiratory failure, Permanent lung damage',
    'Prognosis': 'Generally good with proper management and avoidance of triggers.'
},
{
    'Disease Name': 'Hypertension (High Blood Pressure)',
    'Alternate Names': 'High Blood Pressure, HBP',
    'Overview': 'Hypertension is a condition in which the force of the blood against the artery walls is too high.',
    'Causes': 'Obesity, Sedentary lifestyle, Salt intake, Stress, Genetics',
    'Symptoms': 'Often no symptoms, Severe cases may have headaches, Shortness of breath, Nosebleeds',
    'Diagnosis': 'Blood pressure measurement, Physical examination, Blood tests',
    'Treatment': 'Lifestyle changes, Medications such as ACE inhibitors and diuretics',
    'Prevention': 'Healthy diet, Regular exercise, Reducing salt intake, Managing stress',
    'Complications': 'Heart attack, Stroke, Kidney damage, Vision loss',
    'Prognosis': 'Good with early diagnosis and treatment, but untreated hypertension can lead to serious complications.'
},
{
    'Disease Name': 'Tuberculosis (TB)',
    'Alternate Names': 'TB, Mycobacterium tuberculosis infection',
    'Overview': 'Tuberculosis is a contagious bacterial infection that mainly affects the lungs but can spread to other organs.',
    'Causes': 'Mycobacterium tuberculosis bacteria, Spread through respiratory droplets',
    'Symptoms': 'Persistent cough, Weight loss, Night sweats, Fever',
    'Diagnosis': 'Chest X-ray, Sputum test, Tuberculin skin test',
    'Treatment': 'Long-term antibiotics such as isoniazid, rifampin',
    'Prevention': 'BCG vaccine, Avoiding close contact with TB patients',
    'Complications': 'Lung damage, Spread of infection to other organs, Death if untreated',
    'Prognosis': 'Curable with appropriate antibiotic treatment, but drug-resistant TB can complicate treatment.'
},
{
    'Disease Name': 'Hepatitis B',
    'Alternate Names': 'Hep B, HBV',
    'Overview': 'Hepatitis B is a viral infection that attacks the liver and can cause both acute and chronic disease.',
    'Causes': 'Hepatitis B virus, Spread through contact with infectious body fluids',
    'Symptoms': 'Jaundice, Fatigue, Abdominal pain, Nausea, Dark urine',
    'Diagnosis': 'Blood tests, Liver biopsy',
    'Treatment': 'Antiviral medications, Liver transplant in severe cases',
    'Prevention': 'Hepatitis B vaccine, Safe sex practices, Avoiding shared needles',
    'Complications': 'Liver cirrhosis, Liver cancer, Liver failure',
    'Prognosis': 'Chronic infection can lead to serious liver damage, but vaccines are highly effective in prevention.'
},
{
    'Disease Name': 'Parkinson’s Disease',
    'Alternate Names': 'PD, Paralysis Agitans',
    'Overview': 'Parkinson’s Disease is a neurodegenerative disorder that affects movement.',
    'Causes': 'Genetic mutations, Environmental factors',
    'Symptoms': 'Tremors, Slowed movement, Muscle stiffness, Impaired balance',
    'Diagnosis': 'Neurological exam, Medical history, MRI or PET scans',
    'Treatment': 'Levodopa, Dopamine agonists, Physical therapy, Deep brain stimulation',
    'Prevention': 'No known prevention, but regular exercise and a healthy diet may reduce risk',
    'Complications': 'Difficulty walking, Speech difficulties, Dementia in advanced stages',
    'Prognosis': 'Chronic and progressive, but manageable with treatment.'
}
]

# Convert structured medical data into descriptive strings
medical_text_descriptions = [generate_medical_case(disease) for disease in medical_texts]

# Preprocess texts and generate embeddings
text_embeddings = process_texts(medical_text_descriptions)

# Add text embeddings to ChromaDB collection
text_ingestion_time_taken = add_to_collection(
    text_collection, 
    embeddings=text_embeddings, 
    metadatas=[{"text": description} for description in medical_text_descriptions], 
    ids=[str(i) for i in range(len(medical_text_descriptions))]
)

# Function to query images or text based on a user query
def search_query(query, mode="text"):
    # Generate the query embedding
    start_time = time.time()  # Start time for query
    inputs = clip_processor(text=query, return_tensors="pt", padding=True)
    with torch.no_grad():
        query_embedding = clip_model.get_text_features(**inputs).numpy()
    
    query_embedding = query_embedding.tolist()
    
    # Perform a vector search in the relevant collection
    collection = image_collection if mode == "image" else text_collection
    results = collection.query(query_embeddings=query_embedding, n_results=1)
    query_time = time.time() - start_time  # Measure query time
    
    # Retrieve the result
    if mode == "image":
        result_image_path = results['metadatas'][0][0]['image']
        return Image.open(result_image_path), f"Top match: {result_image_path.split('/')[-1]}", f"Query time: {query_time:.4f} seconds"
    else:
        result_text = results['metadatas'][0][0]['text']
        return result_text, "Matched medical text case", f"Query time: {query_time:.4f} seconds"

# Gradio Interface for text search interaction
def gradio_text_interface(query):
    result, match_info, query_time = search_query(query, mode="text")
    return result, match_info, query_time

# Gradio Interface for image search interaction
def gradio_image_interface(query):
    result, match_info, query_time = search_query(query, mode="image")
    return result, match_info, query_time

# Create a dual interface with Blocks layout
with gr.Blocks() as demo:
    gr.Markdown("# Dual CLIP-based Medical Search Interface")
    
    with gr.Row():
        # Text Search Section
        with gr.Column():
            text_query = gr.Textbox(label="Enter your text query")
            text_output = gr.Textbox(label="Text Search Results", interactive=False)
            text_query_time_output = gr.Textbox(label="Query Time", interactive=False)
            text_button = gr.Button("Search Text")
            
            # Link button to the search function
            text_button.click(gradio_text_interface, inputs=text_query, outputs=[text_output, text_query_time_output])

        # Image Search Section
        with gr.Column():
            image_query = gr.Textbox(label="Enter your image query")
            image_output = gr.Image(type="pil", label="Image Search Results")
            image_text_output = gr.Textbox(label="Image Search Results Text", interactive=False)
            image_query_time_output = gr.Textbox(label="Query Time", interactive=False)
            image_button = gr.Button("Search Image")
            
            # Link button to the search function
            image_button.click(gradio_image_interface, inputs=image_query, outputs=[image_output, image_text_output, image_query_time_output])

# Launch the demo
demo.launch(share=True)
