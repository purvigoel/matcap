import runway
from runway.data_types import number, text, image
from PIL import Image
import matcap_model 
import torch
import numpy

@runway.setup(options={"checkpoint": runway.category(description="Pretrained checkpoints to use.", choices=['skip'], default='skip')})
def setup(opts):
    msg = '[SETUP] Running Model'
    print(msg)
    model = matcap_model.get_generator()
    return model

# Every model needs to have at least one command. Every command allows to send
# inputs and process outputs. To see a complete list of supported inputs and
# outputs data types: https://sdk.runwayml.com/en/latest/data_types.html
@runway.command(name='generate',
                inputs={ 'patch': image(width=16, height=16) },
                outputs={ 'image': image(width=64, height=64) })
def generate(model, args):
    # Generate a PIL or Numpy image based on the input caption, and return it
    img = args['patch']
    img_tensor  = torch.tensor(numpy.array(img)).float() 
    hacky_workaround = []
    hacky_workaround.append(img_tensor)
    for i in range(99):
        hacky_workaround.append(torch.rand(16,16,3))
    noise = torch.stack(hacky_workaround).float()
    noise = noise.unsqueeze(0).view(100, -1).float() 

    input_image = Image.open("./input.png")
    input_image = numpy.array(input_image.resize((16, 16))) / 256.0

    input_images = []

    for inp in range(100):
      input_images.append(input_image)
    input_images = torch.tensor(input_images).float()


    inp = input_images.view(100, -1)
    noise = torch.cat((noise, inp),dim=1).unsqueeze(0).unsqueeze(0).permute(2, 3, 0 ,1)


    output_image = model(noise)[0]
    output_image = output_image.clamp(min=-1, max=1)
    output_image = (output_image - output_image.min())/(output_image.max() - output_image.min())
    # output_image = ((output_image + 1.0) * 255 / 2.0)
    output_image = output_image.permute(1,2,0) * 256.0
    output_image = output_image.detach().numpy().astype(numpy.uint8)
    return {
        'image': output_image
    }

if __name__ == '__main__':
    # run the model server using the default network interface and ports,
    # displayed here for convenience
    runway.run(host='0.0.0.0', port=8000)
