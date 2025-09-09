# AI-acadmic-Advaisor
 pip install -r requirements.txt
 pip install pydantic-ai



added envirment varibales :


tesseract
download: https://github.com/UB-Mannheim/tesseract/wiki
add to envirment varicles under PATH

tesseract
download: https://github.com/UB-Mannheim/tesseract/wiki
add to envirment varicles under PATH


docker run -d --name mongodb -p 27017:27017 -v mongodata:/data/db mongo


#to delete data from chromaDB:
Remove-Item -Recurse -Force .\chroma_db\

#to delete data from ingested_files:
Remove-Item -Force .\ingested_files.json

Remove-Item -Force .\majors_profiles.json

Remove-Item -Force .\majors_embeddings.json
Remove-Item -Recurse -Force .\chroma_db\
Remove-Item -Force .\ingested_files.json
Remove-Item -Force .\extracted_majors.json

