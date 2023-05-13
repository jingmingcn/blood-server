import pytesseract
import cv2
from PIL import Image
import json

if __name__ == '__main__':
    image = cv2.imread("origin_pics/2.jpg")
    text = pytesseract.image_to_string(Image.fromarray(image), lang='chi_sim', config='--psm 1 digits')
    print(text)

    # with open('bloodtestdata.json') as json_file:
    #         data = json.load(json_file)
    #         str = ''
    #         for i in range(22):
    #               for j in range(len(data['bloodtest'][i]['name'])):
    #                     str += data['bloodtest'][i]['name'][j:j+1:1]+'\n'
    #         with open('tmp/t.txt', 'w') as f:
    #             f.write(str)
