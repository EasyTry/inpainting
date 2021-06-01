import io

from model import Inpainter

model = Inpainter(height=1000)

def call(img, mask):
	inpainted = model(img, mask)
	bytes_io = io.BytesIO()
    	inpainted.save(bytes_io, format='JPEG')
    	return bytes_io.getvalue()
