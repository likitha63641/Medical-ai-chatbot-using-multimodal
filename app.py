from flask import Flask, render_template, request, redirect, url_for, session, flash
from tensorflow.keras.models import load_model
import numpy as np
from werkzeug.utils import secure_filename
import os
from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import os
from werkzeug.utils import secure_filename
import PIL
import PIL.Image  #
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from tensorflow.keras.models import load_model
import json
import os
from PIL import Image
import numpy as np
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash

UPLOAD_FOLDER = 'static/uploads/'
app = Flask(__name__)


app.secret_key = 'your_random_secret_key'  



ALLOWED_EXTENSIONS = set(['jpg', 'jpeg','png'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024


def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

from functools import wraps

DB_PATH = "users.db"

# ---------- DB helpers ----------
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db() as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

# ---------- Auth helpers ----------
def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to continue.', 'warning')
            return redirect(url_for('login'))
        return view(*args, **kwargs)
    return wrapped

# ---------- Routes ----------
@app.route('/', methods=['GET', 'POST'])
def login():
    msg = ''
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()

        with get_db() as con:
            row = con.execute(
                "SELECT id, username, password_hash FROM users WHERE username = ?",
                (username,)
            ).fetchone()

        if row and check_password_hash(row['password_hash'], password):
            session['user_id'] = row['id']
            session['username'] = row['username']
            flash('Logged in successfully.', 'success')
            return redirect(url_for('home'))
        else:
            msg = 'Incorrect username/password!'
    return render_template('login.html', msg=msg)

@app.route('/register', methods=['GET', 'POST'])
def register():
    msg = ''
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        confirm  = request.form.get('confirm', '').strip()

        if not username or not password:
            msg = 'Username and password are required.'
        elif password != confirm:
            msg = 'Passwords do not match.'
        else:
            try:
                pw_hash = generate_password_hash(password)
                with get_db() as con:
                    con.execute(
                        "INSERT INTO users (username, password_hash) VALUES (?, ?)",
                        (username, pw_hash)
                    )
                flash('Registration successful. Please log in.', 'success')
                return redirect(url_for('login'))
            except sqlite3.IntegrityError:
                msg = 'Username already taken. Choose another.'
    return render_template('register.html', msg=msg)

# @app.route('/home')
# @login_required
# def home():
#     # Example protected page
#     return f"<h1>Welcome, {session.get('username')}!</h1><p>This is a protected page.</p><p><a href='{url_for('logout')}'>Logout</a></p>"




@app.route('/home')
@login_required
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/disease')
def disease():
    return render_template('disease.html')



disease_info = {
  "Eczema": {
    "emoji": "🧴",
    "symptoms": "Intense itching (often worse at night), red or brownish-gray patches on the skin (especially hands, feet, ankles, wrists, neck, upper chest), small raised bumps that may leak fluid, thickened or scaly skin from chronic scratching.",
    "precautions": "Avoid harsh soaps, detergents, and fragrances. Wear soft cotton clothing. Use humidifiers to prevent dry skin. Take short, warm showers. Manage stress and avoid allergens.",
    "treatment": "Apply thick moisturizers frequently (especially after bathing). Use prescription topical corticosteroids. In severe cases, consider oral antihistamines, immunosuppressants, or biologics like dupilumab (Dupixent)."
  },
  "Warts Molluscum and other Viral Infections": {
    "emoji": "🦠",
    "symptoms": "Warts: small, rough bumps often on hands and feet. Molluscum: dome-shaped, flesh-colored, umbilicated (dimpled) lesions that may spread. Both are contagious and usually painless.",
    "precautions": "Avoid sharing towels, razors, or shoes. Do not scratch or pick lesions. Keep affected areas clean and covered. Practice good hand hygiene.",
    "treatment": "Topical salicylic acid for warts, cryotherapy (freezing), curettage (scraping), or laser treatment. Molluscum can resolve on its own in 6–12 months, but severe cases may need minor surgical removal or topical antivirals like imiquimod."
  },
  "Melanoma": {
    "emoji": "⚠️",
    "symptoms": "A new or changing mole, especially one that is asymmetric, has irregular borders, multiple colors, diameter >6mm, or evolving in shape/size (ABCDE rule). May bleed, itch, or ulcerate.",
    "precautions": "Avoid sunburns and use broad-spectrum SPF 30+ sunscreen daily. Wear protective clothing and avoid tanning beds. Perform monthly self-skin checks and annual dermatologist evaluations if high-risk.",
    "treatment": "Surgical excision with margins is standard. Sentinel lymph node biopsy for staging. Advanced cases may require immunotherapy (e.g., checkpoint inhibitors), targeted therapy (e.g., BRAF inhibitors), or radiation therapy."
  },
  "Atopic Dermatitis": {
    "emoji": "🤧",
    "symptoms": "Dry, scaly, itchy skin that flares periodically. Common in infants and children. Lesions typically occur on cheeks, elbows, knees, and hands. Chronic scratching can lead to lichenification (thick skin).",
    "precautions": "Keep skin well moisturized. Avoid triggers like allergens, overheating, rough fabrics, and emotional stress. Use non-soap cleansers and bathe in lukewarm water.",
    "treatment": "Daily use of emollients. Topical corticosteroids or calcineurin inhibitors (tacrolimus/pimecrolimus). Antihistamines to reduce itch. Severe cases may benefit from systemic therapies or biologics (e.g., dupilumab)."
  },
"Basal Cell Carcinoma": {
    "emoji": "❗",
    "symptoms": "A shiny, pearly, or waxy bump often on the face, ears, or neck. May appear as a flat, flesh-colored or brown scar-like lesion. Slow-growing and rarely spreads but can cause local tissue damage if untreated.",
    "precautions": "Avoid prolonged sun exposure. Use SPF 30+ sunscreen daily. Wear protective clothing and wide-brimmed hats. Get regular dermatology check-ups, especially if you have a history of skin cancer.",
    "treatment": "Surgical excision (Mohs micrographic surgery is gold standard), curettage and electrodessication, cryotherapy. In select cases, topical treatments (imiquimod, fluorouracil) or radiation therapy may be used."
  },
  "Melanocytic Nevi (NV)": {
    "emoji": "🟤",
    "symptoms": "Commonly known as moles. Small, dark, and well-defined spots that may be flat or raised. Usually benign, stable in shape and color.",
    "precautions": "Monitor moles using the ABCDE rule (Asymmetry, Border, Color, Diameter, Evolution). Avoid excessive sun exposure. Report any changes to a dermatologist.",
    "treatment": "No treatment is typically required. Moles may be surgically removed if cosmetically undesired or suspected to be precancerous or malignant."
  },
  "Benign Keratosis-like Lesions": {
    "emoji": "🧊",
    "symptoms": "Seborrheic keratoses: waxy, wart-like growths that can be tan, brown, or black. Often look 'stuck-on' and are common in older adults.",
    "precautions": "No specific prevention. Protect skin from sun damage to reduce lesion development. Regular self-checks can help spot any suspicious growths.",
    "treatment": "Usually no treatment needed unless irritated or for cosmetic reasons. Removal options include cryotherapy (freezing), curettage, electrosurgery, or laser therapy."
  },
  "Psoriasis pictures Lichen Planus and related diseases": {
    "emoji": "🌿",
    "symptoms": "Psoriasis: raised, red patches covered with silvery scales, often on elbows, knees, scalp. Lichen Planus: flat-topped, purple, itchy bumps, commonly on wrists, ankles, or mucous membranes.",
    "precautions": "Avoid skin trauma, stress, and known triggers (infections, certain drugs). Keep skin moisturized. Limit alcohol intake and avoid smoking.",
    "treatment": "Topical corticosteroids, vitamin D analogs, and calcineurin inhibitors. Phototherapy (UVB). Systemic treatments for moderate-to-severe cases (methotrexate, biologics like adalimumab or secukinumab)."
  },
  "Seborrheic Keratoses and other Benign Tumors": {
    "emoji": "🧬",
    "symptoms": "Non-cancerous skin growths that look waxy, stuck-on, or wart-like. Often brown, black, or tan. Common with aging, mostly on chest, back, face, or scalp.",
    "precautions": "Generally no precautions required. Monitor for changes in appearance and avoid irritation from clothing or scratching.",
    "treatment": "Not required unless for cosmetic purposes or if symptomatic. Removal by cryotherapy, curettage, or electrosurgery."
  },
  "Tinea Ringworm Candidiasis and other Fungal Infections": {
    "emoji": "🍄",
    "symptoms": "Tinea: ring-shaped, red, itchy rash with central clearing. Candidiasis: red, moist patches in skin folds. Often found in warm, moist areas like feet, groin, armpits, or under breasts.",
    "precautions": "Keep skin clean and dry. Avoid tight or damp clothing. Don’t share towels, shoes, or personal hygiene items. Use antifungal powder in shoes or sweaty areas.",
    "treatment": "Topical antifungal creams (clotrimazole, terbinafine). Oral antifungals (fluconazole, itraconazole) for widespread or nail/scalp infections. Maintain hygiene and treat recurrences promptly."
  }
}

model_path = 'my_model.h5'
model = load_model(model_path)
print(" Model loaded successfully.")

# --- Load class names ---
class_names_path = 'class_names.json'
with open(class_names_path, 'r') as f:
    class_names = json.load(f)
print(" Class names loaded.")

def load_and_prepare_image(img_path, target_size=(256, 256)):
    img = Image.open(img_path).convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img, dtype=np.float32)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0), img

@app.route('/predict', methods=['POST'])
def predict():
    if 'imagefile' not in request.files:
        return redirect(request.url)

    file = request.files['imagefile']
    if file.filename == '':
        return redirect(request.url)

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    processed_img, original_img = load_and_prepare_image(filepath)

    # Predict
    pred_probs = model.predict(processed_img)
    predicted_index = np.argmax(pred_probs)
    confidence = float(np.max(pred_probs))
    predicted_class = class_names[predicted_index]
    info = disease_info.get(predicted_class, {})


    return render_template(
        'disease.html',
        result=1,
        label=predicted_class,
        emoji=info.get('emoji', '❓'),
        probability=round(confidence * 100, 2),
        symptoms=info.get('symptoms', 'N/A'),
        precautions=info.get('precautions', 'N/A'),
        treatment=info.get('treatment', 'N/A'),
        filepath=filename
    )



import pickle
from keras.models import load_model as load_chatbot_model
from nltk.stem import WordNetLemmatizer
import nltk
import json
import random

# Load chatbot model, data, and intents
chatbot_model = load_chatbot_model('chatbot_model.h5')
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
lemmatizer = WordNetLemmatizer()

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    p = bow(sentence, words, show_details=False)
    res = chatbot_model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def get_response(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            return random.choice(i['responses'])
    return "I’m not sure I understand. Can you rephrase?"

@app.route('/chat', methods=['POST'])
def chat():
    user_msg = request.json.get("message")
    ints = predict_class(user_msg)
    res = get_response(ints, intents)
    return {"response": res}



@app.route('/logout')
def logout():
    session.pop('user_id', None)
    session.pop('username', None)
    flash('Logged out successfully.', 'info')
    return redirect(url_for('login'))





if __name__ == '__main__':
    init_db()
    app.run(debug=True)

