import pandas as pd
import sqlalchemy
import uuid
import hashlib
import os

from fastapi import FastAPI, Depends, HTTPException, File, UploadFile, Form
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from db import connect_unix_socket, connect_tcp_socket
from dotenv import load_dotenv
from sqlalchemy.orm import Session
from pathlib import Path
from sqlalchemy import text
# Import kebutuhan Login
from fastapi.security import OAuth2PasswordBearer
from jose import jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta
# Import Kebutuhan untuk upload to Cloud Storage
from google.cloud import storage

# Load environment variables
dotenv_path = Path("./.env")
load_dotenv(dotenv_path=dotenv_path)

# Define models for register
def generate_password_hash(password):
    hashed_password = pwd_context.hash(password.encode())
    return hashed_password

# Konfigurasi JWT
SECRET_KEY = "maqila"  # Ganti dengan kunci rahasia yang kuat
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Initialize FastAPI app
app = FastAPI()

# Konfigurasi otentikasi
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Konfigurasi password hashing
pwd_context = CryptContext(schemes=["bcrypt", "sha256_crypt"], deprecated="auto")

# Fungsi bantuan untuk membuat token
def create_access_token(data: dict, expires_delta: timedelta or None = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# Fungsi bantuan untuk mendapatkan user dari database berdasarkan username
def get_user(db_session, email: str):
    with db_session.connect() as conn:
        try:
            existing_user = conn.execute(
                sqlalchemy.text(f'SELECT * FROM "user" WHERE "email" = \'{email}\';')
            ).fetchone()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error: {e}")
    return existing_user[3]

# Fungsi bantuan untuk memverifikasi kata sandi
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

#Inisialisasi Bucket
bucket_name = "adoptify-bucket"  # Ganti dengan nama bucket GCS Anda

#Fungsi Upload to Bucket
def upload_to_bucket(blob_name, file_content, bucket_name):
    """Upload data to a Google Cloud Storage bucket."""
    storage_client = storage.Client.from_service_account_json('creds.json')
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_file(file_content)
    return blob.public_url

#Fungsi membuat Filename baru
def generate_new_filename(file, petId):
    """Generate a new filename based on a specific format."""
    file_extension = file.filename.split(".")[-1]  # Get the file extension
    new_filename = f"{petId}_{str(uuid.uuid4())[:8]}.{file_extension}"  # Create a new filename using UUID
    return new_filename

#Fungsi Insert Data
def insert_adoption_record(nama, email, alamat, prov, pos, kartuIdentitas, buktiTransfer, petId):
    """Insert adoption record into PostgreSQL database."""
    db = connect_unix_socket()
    # db = connect_tcp_socket()
    with db.connect() as conn:
        try:
            query = text(
                'INSERT INTO tbAdopt (nama, email, alamat, provinsi, kodepos, ktp, tf, petid) '
                'VALUES (:nama, :email, :alamat, :prov, :pos, :kartuIdentitas, :buktiTransfer, :petId)'
            )
            conn.execute(
                query,
                {
                    "nama": nama,
                    "email": email,
                    "alamat": alamat,
                    "prov": prov,
                    "pos": pos,
                    "kartuIdentitas": kartuIdentitas,
                    "buktiTransfer": buktiTransfer,
                    "petId": petId,
                },
            )
            conn.commit()
        except Exception as e:
            conn.rollback()  # Rollback transaksi jika ada kesalahan
            raise e

# Login API
@app.post("/api/login")
async def login_for_access_token(
    email: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(connect_unix_socket) #connect_tcp_socket
):
    if not email or not password:
        raise HTTPException(status_code=400, detail="Email and password are required")
    hashedPassUser = get_user(db, email)
    if not email or not verify_password(password, hashedPassUser):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(data={"sub": email}, expires_delta=access_token_expires)
    success_message = f"Successfully logged in as {email}"
    return {"message": success_message, "access_token": access_token, "token_type": "bearer"}

# Register API
@app.post("/api/register")
async def register(username: str = Form(...),
                   email: str = Form(...),
                   password:str = Form(...),
                   db: Session = Depends(connect_unix_socket)
                ): #connect_tcp_socket
    # Check if username and email are provided
    if not username or not email:
        raise HTTPException(status_code=400, detail="Missing username or email")

    # Check if email is already registered
    with db.connect() as conn:
        try:
            existing_user = conn.execute(
                sqlalchemy.text(f'SELECT * FROM "user" WHERE "email" = \'{email}\';')
            ).fetchone()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error: {e}")

        if existing_user:
            raise HTTPException(status_code=409, detail="Email already registered")

    # Hash password

    hashed_password = generate_password_hash(password)

    # Insert user data
    with db.connect() as conn:
        try:
            conn.execute(
                sqlalchemy.text(
                    f'INSERT INTO "user" (username, email, password) VALUES '
                    f"('{username}', '{email}', '{hashed_password}')"
                )
            )
            conn.commit()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error: {e}")

    return {
        "status": 201,
        "msg": "Successfully Registered",
        "data": {
            "username": username,
            "email": email,
        },
    }

#API Pet-recommendation
@app.get("/api/pet-recommendations")
async def pet_recommendations(petId: int, recomType: str):
    db = connect_unix_socket()
    # db = connect_tcp_socket()
    with db.connect() as conn:
        data = pd.read_sql(
            sqlalchemy.text(
                "SELECT * FROM tbAdoptify"
            ),
            conn
        )
    # print(data.keys())
    
    data["kontak"] = data["kontak"].str.replace("'", "")
    # data.rename(columns={"ID": "UID"}, inplace=True)
    # RAS

    tf = TfidfVectorizer()
    tf.fit(data["ras"])
    # tf.get_feature_names_out()

    tfidf_matrix_ras = tf.fit_transform(data["ras"])

    cosine_sim_ras = cosine_similarity(tfidf_matrix_ras)
    #cosine_sim_ras

    cosine_sim_df_ras = pd.DataFrame(
        cosine_sim_ras, index=data["uid"], columns=data["uid"]
    )
    # print(cosine_sim_df_ras)


    def ras_hewan_recommendations(
        UID, similarity_data=cosine_sim_df_ras, items=data, k=5
    ):
        index = similarity_data.loc[:, UID].to_numpy().argpartition(range(-1, -k, -1))
        closest = similarity_data.columns[index[-1 : -(k + 2) : -1]]
        closest = closest.drop(UID, errors="ignore")
        return pd.DataFrame(closest).merge(items).head(k)

    # Kesehatan
    tf = TfidfVectorizer()
    tf.fit(data["kesehatan"])


    tfidf_matrix_kesehatan = tf.fit_transform(data["kesehatan"])


    cosine_sim_kesehatan = cosine_similarity(tfidf_matrix_kesehatan)
    cosine_sim_df_kesehatan = pd.DataFrame(
        cosine_sim_kesehatan, index=data["uid"], columns=data["uid"]
    )


    def kesehatan_hewan_recommendations(
        UID,
        similarity_data=cosine_sim_df_kesehatan,
        items=data,
        k=5,
    ):
        index = similarity_data.loc[:, UID].to_numpy().argpartition(range(-1, -k, -1))
        closest = similarity_data.columns[index[-1 : -(k + 2) : -1]]
        closest = closest.drop(UID, errors="ignore")
        return pd.DataFrame(closest).merge(items).head(k)


    # JENIS
    tf = TfidfVectorizer()
    tf.fit(data["jenis"])
    tf.get_feature_names_out()

    tfidf_matrix_jenis = tf.fit_transform(data["jenis"])


    cosine_sim_jenis = cosine_similarity(tfidf_matrix_jenis)
    cosine_sim_df_jenis = pd.DataFrame(
        cosine_sim_jenis, index=data["uid"], columns=data["uid"]
    )

    def jenis_hewan_recommendations(
        UID, similarity_data=cosine_sim_df_jenis, items=data, k=5
    ):
        index = similarity_data.loc[:, UID].to_numpy().argpartition(range(-1, -k, -1))
        closest = similarity_data.columns[index[-1 : -(k + 2) : -1]]
        closest = closest.drop(UID, errors="ignore")
        return pd.DataFrame(closest).merge(items).head(k)

    # MEAN RAS KESEHATAN
    mean_data = (cosine_sim_df_kesehatan + cosine_sim_df_ras) / 2

    def mean_kesehatan_ras_recommendation(
        UID, similarity_data=mean_data, items=data, k=10
    ):
        index = similarity_data.loc[:, UID].to_numpy().argpartition(range(-1, -k, -1))
        closest = similarity_data.columns[index[-1 : -(k + 2) : -1]]
        closest = closest.drop(UID, errors="ignore")
        return pd.DataFrame(closest).merge(items).head(k)
    
    df_result = pd.DataFrame()

    if recomType.lower() == "all":
        df_result = mean_kesehatan_ras_recommendation(petId)
    if recomType.lower() == "ras":
        df_result = ras_hewan_recommendations(petId)
    if recomType.lower() == "kesehatan":
        df_result = kesehatan_hewan_recommendations(petId)

    return {
        "status": 200,
        "msg": "Success Generate Recommendations",
        "data": df_result.to_dict("records"),
    }
    
#API list pet
@app.get("/api/pet")
async def pet():
    db = connect_unix_socket()
    # db = connect_tcp_socket()
    with db.connect() as conn:
        data = pd.read_sql(
            sqlalchemy.text(
                f"SELECT * FROM tbAdoptify"
            ),
            conn
        )
    return {
        "status": 200,
        "msg": "Success Generate Recommendations",
        "data": data.to_dict('records'),
    }
    
#API detail pet
@app.get("/api/pet-detail")
async def pet_detail(petId: int):
    db = connect_unix_socket()
    # db = connect_tcp_socket()
    with db.connect() as conn:
        data = pd.read_sql(
            sqlalchemy.text(
                f"SELECT * FROM tbAdoptify WHERE uid = {petId}"
            ),
            conn
        )
    return {
        "status": 200,
        "msg": "Success Generate Detail Pet",
        "data": data.to_dict('records'),
    }

#API berdasarkan jenis
@app.get("/api/pets-byType")
async def pet_recommendations(petType: str):
    db = connect_unix_socket()
    # db = connect_tcp_socket()
    with db.connect() as conn:
        data = pd.read_sql(
            sqlalchemy.text(
                f"SELECT * FROM tbAdoptify WHERE jenis = '{petType}'"
            ),
            conn
        )
    return {
        "status": 200,
        "msg": "Success Generate Recommendations",
        "data": data.to_dict('records'),
    }

#API berdasarkan ras
@app.get("/api/pets-byRas")
async def pet_recommendations(petType: str):
    db = connect_unix_socket()
    # db = connect_tcp_socket()
    with db.connect() as conn:
        data = pd.read_sql(
            sqlalchemy.text(
                f"SELECT * FROM tbAdoptify WHERE ras = '{petType}'"
            ),
            conn
        )
    return {
        "status": 200,
        "msg": "Success Generate Recommendations",
        "data": data.to_dict('records'),
    }

#API berdasarkan kesehatan
@app.get("/api/pets-byKesehatan")
async def pet_recommendations(petType: str):
    db = connect_unix_socket()
    # db = connect_tcp_socket()
    with db.connect() as conn:
        data = pd.read_sql(
            sqlalchemy.text(
                f"SELECT * FROM tbAdoptify WHERE kesehatan = '{petType}'"
            ),
            conn
        )
    return {
        "status": 200,
        "msg": "Success Generate Recommendations",
        "data": data.to_dict('records'),
    }
    
#API list Shelter
@app.get("/api/shelter")
async def shelter():
    db = connect_unix_socket()
    # db = connect_tcp_socket()
    with db.connect() as conn:
        data = pd.read_sql(
            sqlalchemy.text(
                f"SELECT * FROM tbShelter"
            ),
            conn
        )
    return {
        "status": 200,
        "msg": "Success Generate ALL Shelter",
        "data": data.to_dict('records'),
    }
    
#API detail Shelter
@app.get("/api/shelter-detail")
async def shelter_detail(shelterId: int):
    db = connect_unix_socket()
    # db = connect_tcp_socket()
    with db.connect() as conn:
        data = pd.read_sql(
            sqlalchemy.text(
                f"SELECT * FROM tbShelter WHERE uid = {shelterId}"
            ),
            conn
        )
    return {
        "status": 200,
        "msg": "Success Generate Detail Shelter",
        "data": data.to_dict('records'),
    }
    
#API adopt
@app.post("/api/adopt")
async def create_adopt(
    nama: str = Form(...),
    email: str = Form(...),
    alamat: str = Form(...),
    prov: str = Form(...),
    pos: str = Form(...),
    kartuIdentitas: UploadFile = File(...),
    buktiTransfer: UploadFile = File(...),
    petId: int = Form(...),
):
    # Validasi tipe file
    if not (kartuIdentitas.content_type.startswith("image/") and buktiTransfer.content_type.startswith("image/")):
        raise HTTPException(status_code=400, detail="Hanya gambar yang diperbolehkan.")

    # Generate new filenames
    new_kartu_filename = generate_new_filename(kartuIdentitas, petId)
    new_bukti_filename = generate_new_filename(buktiTransfer, petId)

    # Melakukan penyisipan ke database dan unggahan file
    insert_adoption_record(nama, email, alamat, prov, pos, new_kartu_filename, new_bukti_filename, petId)

    # Unggah file ke Google Cloud Storage dengan nama baru
    kartu_url = upload_to_bucket(f"adopt/{new_kartu_filename}", kartuIdentitas.file, bucket_name)
    bukti_url = upload_to_bucket(f"adopt/{new_bukti_filename}", buktiTransfer.file, bucket_name)

    return {
        "status": 200,
        "msg": "Pengadopsian berhasil",
        "kartuIdentitas_url": kartu_url,
        "buktiTransfer_url": bukti_url,
    }

#API Adoption
@app.get("/api/adopt")
async def adopt():
    db = connect_unix_socket()
    # db = connect_tcp_socket()
    with db.connect() as conn:
        data = pd.read_sql(
            sqlalchemy.text(
                f"SELECT * FROM tbAdopt"
            ),
            conn
        )
    return {
        "status": 200,
        "msg": "Success Generate Detail Adoption",
        "data": data.to_dict('records'),
    }
    
#API detail Adoption
@app.get("/api/adopt-detail")
async def adopt_detail(adoptId: int):
    db = connect_unix_socket()
    # db = connect_tcp_socket()
    with db.connect() as conn:
        data = pd.read_sql(
            sqlalchemy.text(
                f"SELECT * FROM tbAdopt WHERE id = {adoptId}"
            ),
            conn
        )
    return {
        "status": 200,
        "msg": "Success Generate Detail Adoption",
        "data": data.to_dict('records'),
    }
    
#API list Abandoned
@app.get("/api/abandoned")
async def abandoned():
    db = connect_unix_socket()
    # db = connect_tcp_socket()
    with db.connect() as conn:
        data = pd.read_sql(
            sqlalchemy.text(
                f"SELECT * FROM tbAbandoned"
            ),
            conn
        )
    return {
        "status": 200,
        "msg": "Success Generate ALL Abandoned",
        "data": data.to_dict('records'),
    }