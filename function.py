import io

from model import Inpainter

model = Inpainter(height=1000)

def call(img_n_mask):
	inpainted = model(img_n_mask)
	bytes_io = io.BytesIO()
	np.save(bytes_io, inpainted, allow_pickle=False)
	return bytes_io.getvalue()
