from random import randint
from optparse import OptionParser
from PIL import Image, ImageFont, ImageDraw


def get_captcha():
    color = (randint(0, 64), randint(0, 64), randint(0, 64))
    base_x = 10
    base_y = 0
    base_size = 30
    captcha = Image.new('RGBA', (130, 50), (243, 251, 254, 255))
    text = ''
    for i in range(4):
        ch = chr(randint(ord('0'), ord('9')))
        text += ch
        delta_size = randint(-4, 4)
        delta_x, delta_y = randint(-5, 5), randint(-5, 5)
        angle = randint(-30, 30)
        img_char = Image.new('RGBA', (60, 40), (0, 0, 0, 0))
        typeface = ImageFont.truetype(r'C:\Windows\Fonts\arialnb.ttf', base_size + delta_size)
        cvs_char = ImageDraw.Draw(img_char)
        cvs_char.text((0, 0), ch, font=typeface, fill=color)
        img_char = img_char.rotate(angle, Image.BILINEAR, True)
        captcha.paste(img_char, (base_x + i * 30, base_y), img_char)
    captcha.save(text + '.png')

if __name__ == '__main__':
    get_captcha()
