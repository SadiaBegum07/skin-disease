from django.conf import settings
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
import pymysql
import tensorflow as tf
from tensorflow.keras.models import Model
import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import model_from_json, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from django.shortcuts import redirect
from tensorflow.keras.layers import Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import EarlyStopping




from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.applications import ResNet50


def get_gradcam(model, img_array, class_index, output_path):
    """
    Generates Grad-CAM heatmap for a single image and saves the overlay.
    """
    # Use the last convolutional layer in ResNet50
    last_conv_layer = model.get_layer('conv5_block3_out')  # ResNet50 default last conv layer

    # Create a model that maps the input image to the activations of the last conv layer
    grad_model = Model(
        inputs=[model.inputs],
        outputs=[last_conv_layer.output, model.output]
    )

    # Compute gradient of the predicted class w.r.t last conv layer
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)

    # Compute guided gradients
    guided_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Create heatmap
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ guided_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    # Read original image
    img = img_array[0]
    img = ((img + 1.0) * 127.5).astype(np.uint8)  # ResNet preprocess_input reverses to -1..1

    # Resize heatmap to match image
    heatmap = cv2.resize(heatmap.numpy(), (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Overlay
    superimposed_img = cv2.addWeighted(img, 0.7, heatmap_color, 0.3, 0)

    # Save
    cv2.imwrite(output_path, superimposed_img)

# ---------------------------
# GLOBAL SETTINGS
# ---------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "model")

class_labels = [
    'Actinic Keratosis', 'Basal Cell Carcinoma', 'Dermatofibroma',
    'Melanoma', 'Nevus', 'Pigmented Benign Keratosis',
    'Seborrheic Keratosis', 'Squamous Cell Carcinoma', 'Vascular Lesion'
]

# ---------------------------
# BASIC PAGES
# ---------------------------
def index(request):
    return render(request, 'index.html')

def login_page(request):
    return render(request, 'Login.html')

def register_page(request):
    return render(request, 'Register.html')

def disease_prediction_page(request):
    return render(request, 'DiseasePrediction.html')

# ---------------------------
# DISEASE PREDICTION (IMAGE UPLOAD)
# ---------------------------
from tensorflow.keras.models import load_model

def DiseasePredictionAction(request):
    if request.method == 'POST' and request.FILES.get('t1'):

        # -----------------------------
        # Save uploaded image
        # -----------------------------
        uploaded_file = request.FILES['t1']
        fs = FileSystemStorage(location=os.path.join(settings.MEDIA_ROOT, 'uploads'))
        filename = fs.save(uploaded_file.name, uploaded_file)
        full_file_path = os.path.join(settings.MEDIA_ROOT, 'uploads', filename)

        # -----------------------------
        # Read image
        # -----------------------------
        img = cv2.imread(full_file_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # -----------------------------
        # Choose model type
        # -----------------------------
        model_type = request.POST.get('model', 'cnn')  # default CNN

        # -----------------------------
        # Preprocess image
        # -----------------------------
        if model_type == 'resnet':
            img_resized = cv2.resize(img, (224, 224))
            img_array = np.expand_dims(img_resized, axis=0).astype('float32')
            from tensorflow.keras.applications.resnet50 import preprocess_input
            img_array = preprocess_input(img_array)  # scales to -1..1
        else:
            img_resized = cv2.resize(img, (64, 64))
            img_array = np.expand_dims(img_resized, axis=0).astype('float32') / 255.0

        # -----------------------------
        # Load model
        # -----------------------------
        if model_type == 'resnet':
            model_path = os.path.join(MODEL_DIR, 'resnet50_model.keras')
        else:
            model_path = os.path.join(MODEL_DIR, 'cnn_model.h5')

        if not os.path.exists(model_path):
            return render(request, 'DiseasePrediction.html', {'result': f'{model_type.upper()} model not found'})

        classifier = load_model(model_path)

        # -----------------------------
        # Predict
        # -----------------------------
        prediction = classifier.predict(img_array)
        predicted_class_index = np.argmax(prediction)
        predicted_class = class_labels[predicted_class_index]

        # -----------------------------
        # Annotate original image
        # -----------------------------
        output_img = cv2.resize(img, (700, 400))
        cv2.putText(output_img, f'Disease: {predicted_class}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        output_filename = f'output_{filename}'
        output_path = os.path.join(settings.MEDIA_ROOT, 'uploads', output_filename)
        cv2.imwrite(output_path, output_img)
        output_file_url = settings.MEDIA_URL + 'uploads/' + output_filename

        # -----------------------------
        # Grad-CAM (only for ResNet50)
        # -----------------------------
        gradcam_file_url = None
        if model_type == 'resnet':
            last_conv_layer = classifier.get_layer('conv5_block3_out')
            grad_model = Model(inputs=[classifier.inputs], outputs=[last_conv_layer.output, classifier.output])

            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(img_array)
                loss = predictions[:, predicted_class_index]

            grads = tape.gradient(loss, conv_outputs)
            guided_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            conv_outputs = conv_outputs[0]
            heatmap = conv_outputs @ guided_grads[..., tf.newaxis]
            heatmap = tf.squeeze(heatmap)
            heatmap = np.maximum(heatmap, 0)
            heatmap /= np.max(heatmap)

            img_orig = ((img_array[0] + 1.0) * 127.5).astype(np.uint8)
            heatmap = cv2.resize(heatmap.numpy(), (img_orig.shape[1], img_orig.shape[0]))
            heatmap = np.uint8(255 * heatmap)
            heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            superimposed_img = cv2.addWeighted(img_orig, 0.7, heatmap_color, 0.3, 0)

            gradcam_filename = f'gradcam_{filename}'
            gradcam_path = os.path.join(settings.MEDIA_ROOT, 'uploads', gradcam_filename)
            cv2.imwrite(gradcam_path, superimposed_img)
            gradcam_file_url = settings.MEDIA_URL + 'uploads/' + gradcam_filename

        # -----------------------------
        # Cleanup original upload
        # -----------------------------
        os.remove(full_file_path)

        # -----------------------------
        # Render template
        # -----------------------------
        return render(request, 'DiseasePrediction.html', {
            'result': predicted_class,
            'image': output_file_url,
            'gradcam': gradcam_file_url
        })

    return render(request, 'DiseasePrediction.html', {'result': 'No file uploaded'})


# ---------------------------
# CNN MODEL TRAINING
# ---------------------------
def runCNN(request):
    X_path = os.path.join(MODEL_DIR, 'X_balanced.npy')
    Y_path = os.path.join(MODEL_DIR, 'Y_balanced.npy')


    if not os.path.exists(X_path) or not os.path.exists(Y_path):
        return render(request, 'ViewOutput.html', {'data': 'CNN dataset files not found'})

    X = np.load(X_path).astype('float32') / 255
    Y = np.load(Y_path)
    Y = to_categorical(Y)
    # Resize training images to 64x64
    X = np.array([cv2.resize(img, (64, 64)) for img in X])


    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    classifier = Sequential([
        Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(64,64,3)),
        BatchNormalization(),
        MaxPooling2D((2,2)),

        Conv2D(64, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2,2)),

        Conv2D(128, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2,2)),

        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),   # 🔥 prevents overfitting
        Dense(y_train.shape[1], activation='softmax')
    ])

    classifier.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
    )
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True
    )

    datagen.fit(X_train)

    early_stop = EarlyStopping(
    monitor='val_loss',
    patience=8,
    restore_best_weights=True
    )


    classifier.fit(
    datagen.flow(X_train, y_train, batch_size=8),
    epochs=100,
    validation_data=(X_test, y_test),
    callbacks=[early_stop],
    verbose=2
    )



    # Save the full model (recommended)
    classifier.save(os.path.join(MODEL_DIR, 'cnn_model.h5'))

    # Or if you still want weights + JSON:
    # classifier.save_weights(os.path.join(MODEL_DIR, 'cnn_model.weights.h5'))
    # with open(os.path.join(MODEL_DIR, 'cnn_model.json'), 'w') as f:
    #     f.write(classifier.to_json())

    # Return evaluation
    return evaluate_model(request, "CNN Skin Disease Classification", classifier, X_test, y_test)




# ---------------------------
# RESNET50 MODEL
# ---------------------------
# ---------------------------
# RESNET50 MODEL (OPTIMIZED)
# ---------------------------
# ---------------------------
# RESNET50 MODEL (OPTIMIZED FOR SMALL DATASET)
# ---------------------------
def runResNet50(request):
    import os
    import cv2
    import numpy as np
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
    from tensorflow.keras.applications import ResNet50
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.utils import to_categorical
    from sklearn.model_selection import train_test_split
    from sklearn.utils.class_weight import compute_class_weight
    from SkinDiseaseApp.views import evaluate_model  # ensure same file import

    # -----------------------------
    # Paths
    # -----------------------------
    X_path = os.path.join(MODEL_DIR, 'Xres.npy')
    Y_path = os.path.join(MODEL_DIR, 'Yres.npy')
    model_path = os.path.join(MODEL_DIR, 'resnet50_model.keras')

    # -----------------------------
    # Load dataset
    # -----------------------------
    if not os.path.exists(X_path) or not os.path.exists(Y_path):
        return render(request, 'ViewOutput.html', {'data': 'ResNet50 dataset files not found'})

    X = np.load(X_path)
    Y_raw = np.load(Y_path)
    Y = to_categorical(Y_raw)

    # Resize images to 224x224 and normalize
    X_resized = np.array([cv2.resize(img, (224, 224)) for img in X]).astype('float32')
    from tensorflow.keras.applications.resnet50 import preprocess_input
    X_resized = preprocess_input(X_resized)  # converts -1..1

    # -----------------------------
    # Train/test split
    # -----------------------------
    X_train, X_test, y_train, y_test, Y_train_raw, Y_test_raw = train_test_split(
        X_resized, Y, Y_raw, test_size=0.2, random_state=42, stratify=Y_raw
    )

    # -----------------------------
    # Class weights for imbalance
    # -----------------------------
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(Y_train_raw),
        y=Y_train_raw
    )
    class_weights = dict(enumerate(class_weights))

    # -----------------------------
    # Load or create model
    # -----------------------------
    if os.path.exists(model_path):
        model = load_model(model_path)
    else:
        # Load ResNet50 base (pretrained on ImageNet)
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
        
        # Freeze first 100 layers, fine-tune last layers
        for layer in base_model.layers[:100]:
            layer.trainable = False
        for layer in base_model.layers[100:]:
            layer.trainable = True

        # Build top model
        model = Sequential([
            base_model,
            Flatten(),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(y_train.shape[1], activation='softmax')
        ])

        # Compile
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        # Data augmentation
        datagen = ImageDataGenerator(
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            brightness_range=[0.7,1.3]
        )
        datagen.fit(X_train)

        # Early stopping
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        )

        # Train
        model.fit(
            datagen.flow(X_train, y_train, batch_size=16),
            validation_data=(X_test, y_test),
            epochs=100,
            callbacks=[early_stop],
            class_weight=class_weights,
            verbose=2
        )

        # Save model
        model.save(model_path)

    # -----------------------------
    # Evaluate
    # -----------------------------
    return evaluate_model(request, "ResNet50 Skin Disease Classification", model, X_test, y_test)

# ---------------------------
# MODEL EVALUATION
# ---------------------------
def evaluate_model(request, name, model, X_test, y_test):
    preds = np.argmax(model.predict(X_test), axis=1)
    y_true = np.argmax(y_test, axis=1)

    accuracy = accuracy_score(y_true, preds) * 100
    precision = precision_score(y_true, preds, average='macro') * 100
    recall = recall_score(y_true, preds, average='macro') * 100
    f1 = f1_score(y_true, preds, average='macro') * 100

    output = f"""
    <table border=1 align=center>
    <tr><th>Algorithm</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th></tr>
    <tr><td>{name}</td><td>{accuracy:.2f}</td><td>{precision:.2f}</td>
    <td>{recall:.2f}</td><td>{f1:.2f}</td></tr>
    </table>
    """

    cm = confusion_matrix(y_true, preds)
    plt.figure(figsize=(8,8))
    sns.heatmap(cm, annot=True, xticklabels=class_labels, yticklabels=class_labels, fmt='g')
    plt.savefig('SkinDiseaseApp/static/samples/confusion.png')
    plt.close()

    return render(request, 'ViewOutput.html', {'data': output})


# ---------------------------
# USER SIGNUP
# ---------------------------
def Signup(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')  # TODO: hash passwords in production!
        contact = request.POST.get('contact')
        email = request.POST.get('email')
        address = request.POST.get('address')

        con = pymysql.connect(
            host='127.0.0.1',
            user='skindisease_user',
            password='Sadia@123',
            database='skindisease',
            charset='utf8'
        )

        with con:
            cur = con.cursor()
            cur.execute("SELECT username FROM register WHERE username=%s", (username,))
            if cur.fetchone():
                return render(request, 'Register.html', {'data': 'Username already exists'})

            cur.execute(
                "INSERT INTO register VALUES (%s,%s,%s,%s,%s)",
                (username, password, contact, email, address)
            )
            con.commit()

        return render(request, 'Register.html', {'data': 'Signup successful'})

# ---------------------------
# USER LOGIN
# ---------------------------
def UserLogin(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')

        con = pymysql.connect(
            host='127.0.0.1',
            user='skindisease_user',
            password='Sadia@123',
            database='skindisease',
            charset='utf8'
        )

        with con:
            cur = con.cursor()
            cur.execute(
                "SELECT * FROM register WHERE username=%s AND password=%s",
                (username, password)
            )
            if cur.fetchone():
                return render(request, 'UserScreen.html', {'data': f'Welcome {username}'})

        return render(request, 'Login.html', {'data': 'Invalid login details'})
    

def UserLogout(request):
    return redirect('index')
