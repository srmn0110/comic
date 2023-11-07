import requests
import io
from PIL import Image
import google.generativeai as palm
from django.shortcuts import render
from django.http import HttpResponse

import base64



palm.configure(api_key="AIzaSyB1-Cgdf2afMwjEcoFxPe5VU0EuOgsqE5Y")  # Replace "YOUR API KEY" with your actual API key provided by Google

API_URL = "https://api-inference.huggingface.co/models/ogkalu/Comic-Diffusion"
headers = {"Authorization": "Bearer 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"}

def query_huggingface_api(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.content

def process_text(request):
    if request.method == 'POST':
        user_question = request.POST['user_question']
        
        # Generate text using palm API
        generated_text = generate_text(user_question)
        
        # Use Hugging Face API for image generation
        image_bytes = query_huggingface_api({
            "inputs": generated_text
        })
        
        #    # Convert the image bytes to PIL Image
        image = Image.open(io.BytesIO(image_bytes))
        
        #    # Render the image and text response in the template
        # return render(request, 'result.html', {'answer_content': generated_text, 'image': image})
        
        
        # return render(request, 'result.html', {'answer_content': generated_text,'image_generated':image})
        image.save('/Users/psrimanreddy/Documents/react-django/major_trial/nlp_image_app/media/output_image.png')
        image_base64 = base64.b64encode(image.tobytes()).decode()

    # Render the image and text response in the template
        return render(request, 'result.html', {'answer_content': generated_text, 'image_base64': image_base64})

    return render(request, 'index.html')

def generate_text(prompt):
    defaults = {
        'model': 'models/text-bison-001',
        'temperature': 0.7,
        'candidate_count': 1,
        'top_k': 40,
        'top_p': 0.95,
        'max_output_tokens': 1024,
        'stop_sequences': [],
        # 'safety_settings': [
        #     {"category": "HARM_CATEGORY_DEROGATORY", "threshold": 1},
        #     {"category": "HARM_CATEGORY_TOXICITY", "threshold": 1},
        #     {"category": "HARM_CATEGORY_VIOLENCE", "threshold": 2},
        #     {"category": "HARM_CATEGORY_SEXUAL", "threshold": 2},
        #     {"category": "HARM_CATEGORY_MEDICAL", "threshold": 2},
        #     {"category": "HARM_CATEGORY_DANGEROUS", "threshold": 2}
        # ],
    }
    
    response = palm.generate_text(**defaults, prompt=prompt)
    return response.result
