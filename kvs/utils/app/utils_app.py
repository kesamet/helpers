"""
Script containing utility functions for the app.
"""
import base64
import json
import pickle
import re
import uuid
from pathlib import Path

import pandas as pd
import streamlit as st


def uri_encode_path(path, mime="image/png"):
    raw = Path(path).read_bytes()
    b64 = base64.b64encode(raw).decode()
    return f"data:{mime};base64,{b64}"


def add_header(path):
    st.markdown(
        "<img src='{}' class='img-fluid'>".format(uri_encode_path(path)),
        unsafe_allow_html=True,
    )


def get_pdf_display(pdfbytes):
    base64_pdf = base64.b64encode(pdfbytes).decode('utf-8')
    pdf_display = f'<embed src="data:application/pdf;base64,{base64_pdf}" width="700" height="1025" type="application/pdf">'
    return pdf_display


def download_button(object_to_download, download_filename, button_text, pickle_it=False):
    """
    Generates a link to download the given object_to_download.

    Args:
        object_to_download: The object to be downloaded.
        download_filename (str): filename and extension of file. e.g. mydata.csv,
            some_txt_output.txt download_link_text (str): Text to display for download link.
        button_text (str): Text to display on download button (e.g. 'click here to download file')
        pickle_it (bool): If True, pickle file.

    Returns
        (str): the anchor tag to download object_to_download
    """
    if pickle_it:
        try:
            object_to_download = pickle.dumps(object_to_download)
        except pickle.PicklingError as e:
            st.write(e)
            return None

    else:
        if isinstance(object_to_download, bytes):
            pass

        elif isinstance(object_to_download, pd.DataFrame):
            object_to_download = object_to_download.to_csv(index=False)

        # Try JSON encode for everything else
        else:
            object_to_download = json.dumps(object_to_download)

    try:
        # some strings <-> bytes conversions necessary here
        b64 = base64.b64encode(object_to_download.encode()).decode()

    except AttributeError as e:
        b64 = base64.b64encode(object_to_download).decode()

    custom_css, button_id = custom_button_style()
    dl_link = custom_css + f'<a download="{download_filename}" id="{button_id}" href="data:file/txt;base64,{b64}">{button_text}</a><br></br>'
    return dl_link


def logout_button(auth_domain):
    custom_css, button_id = custom_button_style()
    lo_link = custom_css + f'<a id="{button_id}" href="https://{auth_domain}/_oauth/logout" target="_self">Logout</a><br></br>'
    return lo_link


def custom_button_style():
    button_uuid = str(uuid.uuid4()).replace('-', '')
    button_id = re.sub('\d+', '', button_uuid)

    custom_css = f""" 
            <style>
                #{button_id} {{
                    background-color: rgb(255, 255, 255);
                    color: rgb(38, 39, 48);
                    padding: 0.25em 0.38em;
                    position: relative;
                    text-decoration: none;
                    border-radius: 4px;
                    border-width: 1px;
                    border-style: solid;
                    border-color: rgb(230, 234, 241);
                    border-image: initial;

                }} 
                #{button_id}:hover {{
                    border-color: rgb(246, 51, 102);
                    color: rgb(246, 51, 102);
                }}
                #{button_id}:active {{
                    box-shadow: none;
                    background-color: rgb(246, 51, 102);
                    color: white;
                    }}
            </style> """
    return custom_css, button_id


def adjust_container_width(width=1000):
    st.markdown(
        f"""
        <style>
        .reportview-container .main .block-container{{
            max-width: {width}px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def colour_text(notes, color="red"):
    st.markdown(
        f"<span style='color: {color}'>{notes}</span>",
        unsafe_allow_html=True)
