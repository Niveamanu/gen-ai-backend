  
from flask import Flask, redirect, request, session, jsonify
from urllib.parse import quote_plus
import requests
import os
from dotenv import load_dotenv
from flask_cors import CORS
from werkzeug.utils import secure_filename
import pandas as pd
 
import google.generativeai as genai
import PyPDF2
import csv 
from bs4 import BeautifulSoup



app = Flask(__name__)
CORS(app)
load_dotenv()
 

 
app.secret_key = os.urandom(24)

CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
REDIRECT_URI = "https://gen-ai-backend-1.onrender.com/callback"

@app.route("/")
def home():
    return "Welcome to LinkedIn Scraper App!"

@app.route("/login")
def login(): 
    linkedin_auth_url = f"https://www.linkedin.com/oauth/v2/authorization?response_type=code&client_id={CLIENT_ID}&redirect_uri={REDIRECT_URI}&scope=openid%20profile%20email"
    return redirect(linkedin_auth_url)
 
@app.route("/callback")
def callback():

    code = request.args.get("code")
    token_url = "https://www.linkedin.com/oauth/v2/accessToken"
    token_data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": REDIRECT_URI,
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "scope":"openid,profile,email"
    }
    headers = {
  'Content-Type': 'application/x-www-form-urlencoded'
    }
    response = requests.request("GET", token_url, headers=headers, data=token_data)
    access_token = response.json().get("access_token")
    session["access_token"] = access_token
    # redirect("http://localhost:3000/")
    return jsonify({'token':access_token, "session": session })
 
@app.route('/search')
def search():
    access_token = session.get("access_token")
    headers = {"Authorization": f"Bearer {access_token}"}
    profile_url = "https://api.linkedin.com/v2/userinfo"
    response = requests.get(profile_url, headers=headers)
    profile_data = response.json()
    return jsonify({
            "email": profile_data.get("email"),
            "email_verified": profile_data.get("email_verified"),
            "family_name": profile_data.get("family_name"),
            "given_name": profile_data.get("given_name"),
            "locale": profile_data.get("locale"),
            "name": profile_data.get("name"),
            "picture": profile_data.get("picture"),
            "sub": profile_data.get("sub"),
        })

 
@app.route('/searchh')
def searchh():
    # Replace this with the actual LinkedIn API URL
    linkedin_api_url = "https://api.linkedin.com/v2/userinfo"

    # Retrieve the access token from the session
    access_token = session.get("access_token")
    if not access_token:
        return jsonify({"error":access_token}), 400
 
    # Call the LinkedIn API
    headers = {
        "Authorization": f"Bearer {access_token}"  # Include the access token
    }

    try:
        response = requests.get(linkedin_api_url, headers=headers)

        # Check for errors in LinkedIn API response
        if response.status_code != 200:
            return jsonify({"error": f"LinkedIn API returned {response.status_code}"}), response.status_code

        linkedin_data = response.json()

        # Process and return the data as a list of key-value pairs
        result = [{"key": key, "value": value} for key, value in linkedin_data.items()]

        return jsonify(result)  # Convert the list to JSON and return it
    except requests.exceptions.RequestException as e:
        return jsonify({"error": str(e)}), 500

UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')  # Creates an 'uploads' folder in your current working directory
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DATAFRAME'] = None
# Ensure the folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True) 
 
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        for file_name in os.listdir(UPLOAD_FOLDER):
         file_path = os.path.join(UPLOAD_FOLDER, file_name)
         if os.path.isfile(file_path):
          os.remove(file_path)
        file.save(filepath)
        file_extension = os.path.splitext(filepath)[-1].lower()
        try:
            if file_extension == '.csv':
                df = pd.read_csv(filepath)
            elif file_extension in ['.xls', '.xlsx']:
                df = pd.read_excel(filepath)
            elif file_extension == '.txt':
                df = pd.read_csv(filepath, delimiter='\t')
            elif file_extension == '.pdf':
                pdf_reader = PyPDF2.PdfReader(filepath)
                text_data = ""
                for page in pdf_reader.pages:
                    text_data += page.extract_text()
        # Convert extracted text into a DataFrame (simple example)
                df = pd.DataFrame({'Content': [text_data]})
            else:
              raise ValueError(f"Unsupported file type: {file_extension}")
 
            app.config['DATAFRAME'] = df  # Store DataFrame in app config
            print("DataFrame uploaded successfully:", df.head())  # Debugging
            return jsonify({'message': 'File uploaded and processed successfully'})
        except Exception as e:
            app.config['DATAFRAME'] = None  # Reset in case of error
            return jsonify({'error': f'Error reading the Excel file: {str(e)}'}), 500

 
genai.configure(api_key=os.getenv("gemini_api_key"))
model = genai.GenerativeModel('gemini-pro')
chat = model.start_chat(history=[])

 
def initialize_chat():
    chat = model.start_chat(history=[])  # Initialize with an empty history
    return chat

 

@app.route('/query', methods=['POST'])
def query():
    # Get the user's query from the request
    user_query = request.json.get('query')

    # Get the uploaded DataFrame from the app config
    df = app.config.get('DATAFRAME')

    # If the DataFrame is available, preprocess it and add relevant info to the chat history
    chat_history = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_query}
    ]

    if df is not None:
        # Preprocess the DataFrame and add relevant info (e.g., the first few rows, column names, etc.)
        data_preview = df.head().to_dict()  # Just an example, you can customize this
        chat_history.append({"role": "assistant", "content": f"The uploaded data: {data_preview}"})
    
    # Initialize the chat if not already done
    if 'chat' not in app.config:
        chat = genai.GenerativeModel('gemini-pro').start_chat(history=[])
        app.config['chat'] = chat
    else:
        chat = app.config['chat']
    
    # Send the chat history to the model and get a response
    response = chat.chat(messages=chat_history)

    return jsonify({'response': response['content']}) 

def process_file(file_path="uploads"):
    _, extension = os.path.splitext(file_path)
    if extension.lower() == '.pdf':
        return process_pdf(file_path)
    else:
        return process_text(file_path)

def process_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def process_pdf(file_path):
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ''
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
UPLOAD_FOLDER_ = 'uploads'
JOBS_FOLDER = 'JOBs'

# Ensure these directories exist
os.makedirs(UPLOAD_FOLDER_, exist_ok=True)
os.makedirs(JOBS_FOLDER, exist_ok=True)
@app.route("/check-file", methods=["GET"])
def check_file():
    file_path = 'JOBs/linkedin-jobs.csv'
    if os.path.exists(file_path):
        return jsonify({"fileExists": True})
    return jsonify({"fileExists": False})
    
@app.route('/predict', methods=['POST'])
def predict():
     
    file_data_dir = UPLOAD_FOLDER_
     
    files = os.listdir(file_data_dir)
     
    file_path = os.path.join(file_data_dir, files[0])
    content = process_file(file_path)
     
# Create the CSV agent with Langchain
     
    question = request.form['question']
     
    response = model.generate_content([question, content])

    return jsonify({'result': response.text})
 
@app.route('/predictForJob', methods=['POST'])
def predictForJob():
     
    file_data_dir = JOBS_FOLDER
     
    files = os.listdir(file_data_dir)
     
    file_path = os.path.join(file_data_dir, files[0])
    content = process_file(file_path)
     
# Create the CSV agent with Langchain
     
    question = request.form['question']
     
    response = model.generate_content([question, content])

    return jsonify({'result': response.text})
 


def linkedin_scraper(webpage, start_index, max_pages):
    if not os.path.exists('JOBs'):
        os.makedirs('JOBs')

    # Path for the CSV file to save job data
    file_path = 'JOBs/linkedin-jobs.csv'
    
    # If file exists, remove it to start fresh
    if os.path.exists(file_path):
        os.remove(file_path)

    with open(file_path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Title', 'Company', 'Location', 'Apply'])
        
        # Loop through the pages based on start_index
        for page_number in range(start_index, start_index + max_pages * 25, 25):  # Increment by 25 for each page
            next_page = webpage + str(page_number)
            print(f"Fetching page: {next_page}")
            
            # Send request to LinkedIn and get the HTML
            response = requests.get(next_page)
            soup = BeautifulSoup(response.content, 'html.parser')

            # Find job listings on the page
            jobs = soup.find_all('div', class_='base-card relative w-full hover:no-underline focus:no-underline base-card--link base-search-card base-search-card--link job-search-card')
            
            for job in jobs:
                job_title = job.find('h3', class_='base-search-card__title').text.strip()
                job_company = job.find('h4', class_='base-search-card__subtitle').text.strip()
                job_location = job.find('span', class_='job-search-card__location').text.strip()
                job_link = job.find('a', class_='base-card__full-link')['href']

                # Write job details to CSV
                writer.writerow([
                    job_title.encode('utf-8'),
                    job_company.encode('utf-8'),
                    job_location.encode('utf-8'),
                    job_link.encode('utf-8')
                ])

            print(f"Page {page_number} data fetched and saved.")

            # Delay to avoid overwhelming LinkedIn's servers (you can adjust the delay if needed)
            # time.sleep(2)  # Uncomment if necessary for throttling requests
            
        print('Data update completed.')
    
    print('File closed.')
    return jsonify({"status": "success"})

@app.route('/scrapes', methods=['POST'])
def Scrape():
    keywords = request.json.get('keywords')
    location = request.json.get('location')
    max_pages = request.json.get('max_pages')

    # Construct webpage URL for job search query
    webpage = f'https://www.linkedin.com/jobs-guest/jobs/api/seeMoreJobPostings/search?keywords={keywords}&location={location}&geoId={location}&trk=public_jobs_jobs-search-bar_search-submit&position=1&pageNum=0&start='
    
    # Start the scraper with initial start_index 0
    return linkedin_scraper(webpage, 0, max_pages)

folder_path_For_Scraped_data="JOBs"
def read_csv_and_send_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        rows = list(reader)  # Store all rows as a list of dictionaries
        
        # Clean up byte strings and ensure they are decoded properly
        for row in rows:
            for key, value in row.items():
                if isinstance(value, bytes):
                    row[key] = value.decode('utf-8')  # Decode bytes to string
                elif isinstance(value, str) and value.startswith("b'"):
                    row[key] = value[2:-1]  # Remove the b' prefix and the ending quote
        return rows


@app.route('/get-jobs', methods=['GET'])
def get_jobs():
    files = os.listdir(folder_path_For_Scraped_data)
    if not files:
        return jsonify({'message': 'No jobs available yet'}), 404

    latest_file = sorted(files)[-1]  # Get the most recent file (assuming it's ordered alphabetically)
    file_path = os.path.join(folder_path_For_Scraped_data, latest_file)
    
    jobs_data = read_csv_and_send_data(file_path)
    return jsonify(jobs_data)
   
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)

 
