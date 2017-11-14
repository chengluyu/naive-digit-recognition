import requests
from PIL import Image
from io import BytesIO

if __name__ == '__main__':
    url = 'http://58.194.172.92:85/api.php/check'
    res = requests.get(url)
    im = Image.open(BytesIO(res.content))
    # im = im.convert('1')
    im.save('captcha.png')
