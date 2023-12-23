from django.shortcuts import render
from .forms import ImageUploadForm
from .models import UploadedImage
from django.shortcuts import render, redirect
from .predict import predict_age
from PIL import Image
import cv2
import numpy as np

# Create your views here.
def main(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_image=form.save()
            image=cv2.imread(uploaded_image.image.path)
            #image_array=np.array(image)
            
            # Perform face detection and cropping (using OpenCV as an example)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray_array=np.array(gray)
            #gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray_array, scaleFactor=1.3, minNeighbors=5)
            if len(faces) > 0:
                x, y, w, h = faces[0]
                cropped_face = gray_array[y:y+h, x:x+w]
                cropped_image = Image.fromarray(cropped_face)

                # Save or display the cropped image
                # For example, you can save it or send it to the template for display
                cropped_image_path = uploaded_image.image.path
                cropped_image.save(cropped_image_path)
            return redirect('result', image_id=form.instance.id)
    else:
        form = ImageUploadForm()

    return render(request, 'myapp/main.html', {'form': form})

def result(request, image_id):
    image = UploadedImage.objects.get(id=image_id)
    image_path=image.image.path
    r_class, confidence = predict_age(image_path)

    return render(request, 'myapp/result.html', {'image':image,'r_class':r_class, 'confidence':confidence})
