import streamlit as st
import pandas as pd
import numpy as np
import pickle
from heapq import nsmallest



# for plotting swatches and images
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from skimage.color import rgb2lab, deltaE_cie76
from bokeh.models.widgets import ColorPicker
from bokeh.io import output_file, show


def make_swatch(r,g,b, array=False):
    '''
    Input: array of RGB-valued pixels in shape (n x m x 3)
    
    Output: PIL image
    ''' 
    new_pixels = np.zeros([105, 105, 3], dtype=np.uint8)
    new_pixels[:,:,0] = r
    new_pixels[:,:,1] = g
    new_pixels[:,:,2] = b
    if array:
        return new_pixels
    if not array:
        new_im = Image.fromarray(new_pixels, 'RGB')
        return new_im


def map_to_dict(eye_color, hair_color, skin_tone, eye_dict, hair_dict, skin_dict):

    # Map to user characterisitcs to saved dictionaries
    eye_dict_inv = dict(map(reversed, eye_dict.items()))
    eye_color_u = eye_dict_inv[eye_color]

    hair_dict_inv = dict(map(reversed, hair_dict.items()))
    hair_color_u = hair_dict_inv[hair_color]

    skin_dict_inv = dict(map(reversed, skin_dict.items()))
    skin_tone_u = skin_dict_inv[skin_tone]

    return eye_color_u, hair_color_u, skin_tone_u


def rgb_to_lab(r, g, b):
  rgb_array = np.zeros([1, 1, 3], dtype=np.uint8)
  rgb_array[:,:,0] = r
  rgb_array[:,:,1] = g
  rgb_array[:,:,2] = b

  # convert input color into LAB
  lab = rgb2lab(rgb_array)
  return lab


def find_closest_colors(df, lab_u, n_match):

  diffs = []

  reds = df.red
  greens = df.green
  blues = df.blue

  for ii in range(len(df)):
      r = reds.iloc[ii]
      g = greens.iloc[ii]
      b = blues.iloc[ii]

      rgb_array = np.zeros([1, 1, 3], dtype=np.uint8)

      rgb_array[:,:,0] = r
      rgb_array[:,:,1] = g
      rgb_array[:,:,2] = b

      lab = rgb2lab(rgb_array)
      diff = deltaE_cie76(lab, lab_u)
      diffs.append(diff)


  # Find n_match closest matches
  diff_best = nsmallest(n_match, np.unique(diffs))
  diff_best.sort()


  idxs = []
  for ii in range(n_match):
    closest_idx = np.where(diffs == diff_best[ii])[0][0]
    idxs.append(closest_idx)

  # closest_match = np.where(diffs == np.min(diffs))[0]

  match_df = df.iloc[idxs]

  return match_df


def rect_with_rounded_corners(image, r, t, c):
    """
    :param image: image as NumPy array
    :param r: radius of rounded corners
    :param t: thickness of border
    :param c: color of border
    :return: new image as NumPy array with rounded corners
    """

    c += (255, )

    h, w = image.shape[:2]

    # Create new image (three-channel hardcoded here...)
    new_image = np.ones((h+2*t, w+2*t, 4), np.uint8) * 255
    new_image[:, :, 3] = 0

    # Draw four rounded corners
    new_image = cv2.ellipse(new_image, (int(r+t/2), int(r+t/2)), (r, r), 180, 0, 90, c, t)
    new_image = cv2.ellipse(new_image, (int(w-r+3*t/2-1), int(r+t/2)), (r, r), 270, 0, 90, c, t)
    new_image = cv2.ellipse(new_image, (int(r+t/2), int(h-r+3*t/2-1)), (r, r), 90, 0, 90, c, t)
    new_image = cv2.ellipse(new_image, (int(w-r+3*t/2-1), int(h-r+3*t/2-1)), (r, r), 0, 0, 90, c, t)

    # Draw four edges
    new_image = cv2.line(new_image, (int(r+t/2), int(t/2)), (int(w-r+3*t/2-1), int(t/2)), c, t)
    new_image = cv2.line(new_image, (int(t/2), int(r+t/2)), (int(t/2), int(h-r+3*t/2)), c, t)
    new_image = cv2.line(new_image, (int(r+t/2), int(h+3*t/2)), (int(w-r+3*t/2-1), int(h+3*t/2)), c, t)
    new_image = cv2.line(new_image, (int(w+3*t/2), int(r+t/2)), (int(w+3*t/2), int(h-r+3*t/2)), c, t)

    # Generate masks for proper blending
    mask = new_image[:, :, 3].copy()
    mask = cv2.floodFill(mask, None, (int(w/2+t), int(h/2+t)), 128)[1]
    mask[mask != 128] = 0
    mask[mask == 128] = 1
    mask = np.stack((mask, mask, mask), axis=2)

    # Blend images
    temp = np.zeros_like(new_image[:, :, :3])
    temp[(t-1):(h+t-1), (t-1):(w+t-1)] = image.copy()
    new_image[:, :, :3] = new_image[:, :, :3] * (1 - mask) + temp * mask

    # Set proper alpha channel in new image
    temp = new_image[:, :, 3].copy()
    new_image[:, :, 3] = cv2.floodFill(temp, None, (int(w/2+t), int(h/2+t)), 255)[1]

    return new_image


# Display logo
logo = Image.open('images/logo_1.png')
st.image(logo, use_column_width=True)
st.header('&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;Makeup dupes uniquely for you')

st.sidebar.header('What kind of product?')
products = ['blush', 'lipstick', 'lip gloss', 'lip pencil', 'bronzer']
product_u = st.sidebar.multiselect('Product', products)


st.sidebar.header('Any particular brand?')
brands = ['No preference','Aether Beauty', 'Anastasia Beverly Hills',
            'Armani Beauty', 'Artist Couture', 'Becca Cosmetics', 
            'Bite Beauty', 'Black Up', 'Bobbi Brown', 'Charlotte Tilbury', 
            'Christian Louboutin', 'Ciaté London', 'Clinique', 'Dior', 
            'Dominique Cosmetics', 'Fenty Beauty by Rihanna', 'Givenchy', 
            'Grande Cosmetics', 'Gucci', 'Guerlain', 'Hourglass', 'Huda Beauty', 
            'INC.redible', 'IT Cosmetics', 'Ilia', 'Jouer Cosmetics', 
            'KVD Vegan Beauty', 'Kaja', 'Kevyn Aucoin', "Kiehl's since 1851", 
            'Kilian', 'Kosas', 'Lancôme', 'Laura Mercier', 'Lawless', 'Lilah B.', 
            'MILK Makeup', 'Make Up For Ever', 'Marc Jacobs Beauty', 'Melt Cosmetics', 
            'NARS', 'Natasha Denona', 'Nudestix', 'Pat McGrath Labs', 'Perricone MD', 
            'Pretty Vulgar', 'RMS Beauty', 'Sephora Collection', 'Shiseido', 'Smashbox', 
            'Stellar', 'SurrattBeauty', 'Tarte', 'Tom Ford', 'Too Faced', 'Trèstique', 
            'Urban Decay', 'Yves Saint Laurent', 'bareMinerals']

brands_u = st.sidebar.multiselect('Brand', brands)

if len(product_u) > 1:
    st.write('Start with one product for now, please!')
elif len(product_u) == 1:
    st.subheader('Ok, let\'s look at blushes.\n')
    st.subheader('Now tell me a bit about yourself.')



palette_imgs = []
palette_captions = []
st.sidebar.header('Which best describes your skin tone?')
# skin_slider = st.sidebar.slider('Shade', min_value = 0, max_value = 8, step=1)  # this is a widget


skin_img_dict = {
				'Porcelain': 'images/porcelain.png', 
				'Fair': 'images/fair.png', 
				'Light': 'images/light.png', 
				'Medium': 'images/medium.png', 
				'Tan': 'images/tan.png', 
				'Olive': 'images/olive.png', 
				'Deep': 'images/deep.png', 
				'Dark': 'images/dark.png', 
				'Ebony': 'images/ebony.png'
				}

skin_menu = st.sidebar.multiselect('Skin Tone', list(skin_img_dict.keys()))

# check that eye_menu only contains one item
if len(skin_menu) > 1:
    st.write('Please choose the closest skin tone.')
elif len(skin_menu) == 1:
    skin_tone_u = skin_menu[0]
    palette_imgs.append(skin_img_dict[skin_tone_u])
    palette_captions.append(skin_tone_u)



eye_img_dict = {
					'Blue':'images/blue.png', 
					'Brown':'images/brown.png', 
					'Green': 'images/green.png', 
					'Grey':'images/gray_eye.png', 
					'Hazel':'images/hazel.png'
					 }

eye_menu = st.sidebar.multiselect('Eye Color', list(eye_img_dict.keys()))


# check that eye_menu only contains one item
if len(eye_menu) > 1:
    st.write('Please choose the most representative eye color.')
elif len(eye_menu) == 1:
    eye_color_u = eye_menu[0]
    palette_imgs.append(eye_img_dict[eye_color_u])
    palette_captions.append(eye_color_u)
    # st.image(eye_img_dict[eye_color_u], caption=eye_color_u)


hair_img_dict = {
						'Blonde': 'images/blonde.png',
						'Brunette': 'images/brunette.png',
						'Auburn': 'images/auburn.png',
						'Black': 'images/black.png',
						'Red': 'images/red.png',
						'Grey': 'images/grey_hair.png'
						}


hair_menu = st.sidebar.multiselect('Hair Color', list(hair_img_dict.keys()))

# check that eye_menu only contains one item
if len(hair_menu) > 1:
    st.write('Please choose the most representative eye color.')
elif len(hair_menu) == 1:
    hair_color_u = hair_menu[0]
    palette_imgs.append(hair_img_dict[hair_color_u])
    palette_captions.append(hair_color_u)
    # st.image(hair_img_dict[hair_color_u], caption=hair_color_u)


# image_iterator = paginator("Select a sunset page", palette_imgs)
# indices_on_page, images_on_page = map(list, zip(*image_iterator))
# print(map(list, zip(*image_iterator)))
st.image(palette_imgs, width=100, caption=palette_captions)


st.sidebar.header('Any ideas about color?')
# color_picker = ColorPicker(color="#ff4466", width=75, height=50)
# st.sidebar.bokeh_chart(color_picker)
st.sidebar.button('Surprise me!')
st.sidebar.subheader('or slide on.')
red_u = st.sidebar.slider('Red', min_value=0, max_value=255)
green_u = st.sidebar.slider('Green', min_value=0, max_value=255)
blue_u = st.sidebar.slider('Blue', min_value=0, max_value=255)

st.subheader('You selected this color:')
swatch = make_swatch(red_u, green_u, blue_u, array=True)
rounded_swatch = rect_with_rounded_corners(swatch, 10, 5, (255, 255, 255))
st.image(rounded_swatch)


def check_inputs():
    try:
        len(brands_u)
        return
    except:
        st.write('I stopped. Please select a brand, even if it\'s "No preference"')
        break
    try: 
        len(skin_tone_u)
        return
    except:
        st.write('I stopped. Please select skin tone and try again')  
        break
    try: 
        len(eye_color_u)
        return
    except:
        st.write('I stopped. Please select eye color and try again!')
        break
    try:
        len(hair_color_u)
        return
    except:
        st.write('I stopped. Please select hair color and try again!')
        break


go_button = st.sidebar.button('Go')
# reset_button = st.sidebar.button('Reset')

if go_button:
    st.header('I\'m going!')

    # check if there is input for everything
    check_inputs()

    # load relevant dataframe
    meta_df = pd.read_csv('dataframes/meta_{}.csv'.format(product_u[0]))
    df = pd.read_csv('dataframes/{}_df.csv'.format(product_u[0]))

    # load model 
    loaded_model = pickle.load(open('models/{}_rf.sav'.format(product_u[0]), 'rb'))

    eye_dict = {0: 'Blue', 1: 'Brown', 2: 'Grey', 3: 'Green', 4: 'Hazel'}
    hair_dict = {0: 'Auburn', 1: 'Black', 2: 'Blonde', 3: 'Brunette', 4: 'Grey', 5: 'Red'}
    skin_dict = {0: 'Dark', 1: 'Deep', 2: 'Ebony', 3: 'Fair', 4: 'Light', 
            5: 'Medium', 6: 'Olive', 7: 'Porcelain', 8: 'Tan'}

    eye_mapped, hair_mapped, skin_mapped = map_to_dict(eye_color_u, hair_color_u, 
                                             skin_tone_u, eye_dict, hair_dict, skin_dict)


    # find closest color matches
    lab_u = rgb_to_lab(red_u, green_u, blue_u)
    n_match = 3
    matches = find_closest_colors(df, lab_u, n_match)

    predicted_stars = np.zeros(n_match)
    for ii in range(len(matches)):
      r = matches.iloc[ii].red
      g = matches.iloc[ii].green
      b = matches.iloc[ii].blue
      predicted_stars[ii] = loaded_model.predict([[eye_mapped,hair_mapped, skin_mapped, r, g, b]]) 


    # find top 3 matches
    idxs_five = np.where(predicted_stars == 5)[0]

    best_matches = matches.iloc[idxs_five]

    st.subheader('Your best matches:')
    swatches = []
    captions = []
    links = []

    for ii in range(len(best_matches[:3])):
        swatch = make_swatch(matches.iloc[ii].red, matches.iloc[ii].green, matches.iloc[ii].blue, array=True)
        new_swatch = rect_with_rounded_corners(swatch, 10, 5, (255, 255, 255))
        swatches.append(new_swatch)
        product_name_ = matches.product_name.iloc[ii]
        review_color_ = matches.review_color.iloc[ii]
        link = meta_df.product_link[meta_df['product_name'] == product_name_].unique()[0]
        captions.append('{0}: {1}'.format(matches.product_name.iloc[ii], matches.review_color.iloc[ii]))
        links.append(link)

    st.image(swatches, width=100, caption=captions)

    st.subheader('Product Links:')
    for jj, link in enumerate(links):
        st.markdown('[{0}]({1})'.format(captions[jj],link))
