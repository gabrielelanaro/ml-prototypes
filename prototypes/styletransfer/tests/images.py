from prototypes.styletransfer.images import load_image

def test_load_image():

    data = load_image("img_lights.jpg")
    
    assert isinstance(data, np.ndarray)
    assert data.shape == (512, 512, 3)


