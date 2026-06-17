import uuid
from pathlib import Path

class DocumentService:

    def __init__(self, processor, upload_dir: str):
        self.processor = processor
        self.upload_dir = Path(upload_dir)

    async def upload_document(self, file):

        # safe filename
        filename = (f"{uuid.uuid4()}_{file.filename}")

        file_path = (self.upload_dir / filename)

        # save file
        with open(file_path, "wb") as f:
            f.write(await file.read())

        return {
            "filename": filename,
            "path": str(file_path)
        }

    async def process_document(self, file_path: str):
        return await self.processor.process_document(file_path)
    

        # file_path = Path(file.filename).name

    # with open(file_path, "wb") as f:
    #     f.write(await file.read())
    
    # background_tasks.add_task(processor.process_document, file_path)
    
    # return UploadResponse(filename=file.filename, status="processing")