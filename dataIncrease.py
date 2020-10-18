#encoding=utf-8

from PIL import Image
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage



baseImageFolder = "F:\maxwong\\AIcompete\\firstStage\\train\\images\\" 
baseLabelFolder = "F:\maxwong\\AIcompete\\firstStage\\train\\labels\\" 
outputImageFolder = "F:\maxwong\\AIcompete\\firstStage\\train\\prePImages\\preP"
outputLabelFolder = "F:\maxwong\\AIcompete\\firstStage\\train\\prePLabels\\"

# the function need to achieve  gauss blur ， eagesharpen 
# the sharpen process is geting the eage of image and add it with the origin image

ia.seed(1)


def processBatchImages(startImgNum,bacthSize):
    images = []
    for i in range(1, bacthSize + 1):
        imageNum = startImgNum + i 
        image = image = ia.imageio.imread(baseImageFolder + str(imageNum) + ".tif")
        images.extend(image)

    # ia.imshow(bbs.draw_on_image(image))

    seq = iaa.Sequential([

        # GuassianBlur only for half of all the images
        iaa.GaussianBlur(sigma=(0.5,0.7)),
        # sharpen images eages,
        iaa.Sharpen(alpha=(0.6,1), lightness=(0.9, 1.1)),

    
        # Add a value of -10 to 10 to each pixel.
        iaa.Add((-20, 20), per_channel=0.5),

        # Change brightness of images (50-150% of original value).
        iaa.Multiply((0.7, 1.3), per_channel=0.5)
        
        
        # adjust the image contrast
        # iaa.GammaContrast((0.5, 1.5), per_channel=True)

    ])

    image_aug = seq(images = images)
    for i  in range(0, bacthSize):
       imageNum = startImgNum + i + 1
       image_aug[i].save(outputImageFolder + str(imageNum) + ".tif") 


# use 40 batchs to finish all the images,each batch include 50 images
batchSize = 5
for i in range(0, 5):
    startNum = i * batchSize
    processBatchImages(startNum, batchSize)
    
