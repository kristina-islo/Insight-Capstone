import streamlit as st
import pandas as pd
import numpy as np

from PIL import Image

from bokeh.models.widgets import ColorPicker
from bokeh.io import output_file, show
from bokeh.models.widgets import Dropdown


# Display logo
logo = Image.open('images/logo_1.png')
st.image(logo, use_column_width=True)

st.sidebar.header('What kind of product?')
products = ['blush', 'lipstick', 'lip gloss', 'lip pencil', 'bronzer']
st.sidebar.multiselect('Product', products)



st.sidebar.header('Any ideas about color?')
color_picker = ColorPicker(color="#ff4466", width=75, height=50)
st.sidebar.bokeh_chart(color_picker)

st.sidebar.header('Which best describes your skin tone?')
skin_tones = ['Porcelain', 'Fair', 'Light', 'Medium', 'Tan',
				'Olive', 'Deep', 'Dark', 'Ebony']
# streamlit.slider(label, min_value=None, max_value=None, value=None, step=None, format=None)

skin_slider = st.sidebar.slider('Shade', min_value = 0, max_value = 8, step=1)  # this is a widget


skin_tone_imgs = ['images/porcelain.png', 'images/fair.png', 'images/light.png', 'images/medium.png', 
					'images/tan.png', 'images/olive.png', 'images/deep.png', 'images/dark.png', 'images/ebony.png']

skin_labels = ['Porcelain', 'Fair', 'Light', 'Medium', 'Tan', 'Olive', 'Deep', 'Dark', 'Ebony']

if skin_slider == 0:
	st.write('You selected:')
	st.image(skin_tone_imgs[0], caption=skin_labels[0])
elif skin_slider == 1:
	st.write('You selected:')
	st.image(skin_tone_imgs[1], caption=skin_labels[1])
elif skin_slider == 2:
	st.write('You selected:')
	st.image(skin_tone_imgs[2], caption=skin_labels[2])
elif skin_slider == 3:
	st.write('You selected:')
	st.image(skin_tone_imgs[3], caption=skin_labels[3])
elif skin_slider == 4:
	st.write('You selected:')
	st.image(skin_tone_imgs[4], caption=skin_labels[4])
elif skin_slider == 5:
	st.write('You selected:')
	st.image(skin_tone_imgs[5], caption=skin_labels[5])
elif skin_slider == 6:
	st.write('You selected:')
	st.image(skin_tone_imgs[6], caption=skin_labels[6])
elif skin_slider == 7:
	st.write('You selected:')
	st.image(skin_tone_imgs[7], caption=skin_labels[7])
elif skin_slider == 8:
	st.write('You selected:')
	st.image(skin_tone_imgs[8], caption=skin_labels[8])



eye_color_imgs = ['images/blue.png', 'images/brown.png', 'images/green.png', 
					'images/grey_eye.png', 'images/hazel.png']
eye_colors = ['Blue', 'Brown', 'Green', 'Gray', 'Hazel']
eye_menu = st.sidebar.multiselect('Eye Color', eye_colors)

if eye_menu == 'Blue':
	st.image(eye_color_imgs[0], caption=eye_colors[0])
elif eye_menu == 'Brown':
	st.image(eye_color_imgs[1], caption=eye_colors[1])
elif eye_menu == 'Green':
	st.image(eye_color_imgs[2], caption=eye_colors[2])
elif eye_menu == 'Gray':
	st.image(eye_color_imgs[3], caption=eye_colors[3])
elif eye_menu == 'Hazel':
	st.image(eye_color_imgs[4], caption=eye_colors[4])


hair_colors = ['Blonde', 'Brunette', 'Auburn', 'Black', 'Red', 'Gray']
st.sidebar.multiselect('Hair Color', hair_colors)


go_button = st.sidebar.button('Go')
reset_button = st.sidebar.button('Reset')

if go_button:
	st.write('I\'m going!')


# st.header('Hair color')
# blonde = st.checkbox("Blonde")
# brunette = st.checkbox("Brunette")
# auburn = st.checkbox("Auburn")
# black = st.checkbox("Black")
# red = st.checkbox("Red")
# grey_hair = st.checkbox("Gray")