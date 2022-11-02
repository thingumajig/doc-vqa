import streamlit as st
import fitz  # import the bindings
from PIL import Image

import torch
import re
from transformers import DonutProcessor, VisionEncoderDecoderModel

st.set_page_config('Document Visual Question Answering', layout='centered', page_icon='🌠')

if 'page_num' not in st.session_state:
    st.session_state.page_num = 0
    st.session_state.block_saved_queries = True

saved_queries = [
            'What is the symbols at the bottom of trading symbols?',
            'What is the address at the top of address of principal executive offices?',
            'What is the name of each exchange on which registered?',
            # 'What is the city at the top of address of principal executive offices?',
            'What is the address of principal executive offices mentioned here?',
            'What is the city in the address of principal executive offices mentioned here?',
            'What is the name at the top of exact name of registrant?',
            "What is the registrant's telephone number?",
            # 'What is the number at the top of Zip Code?',
            'What is the Zip Code?',

            'What is the percent of the interest in Pacificor?',
            'What is the name of the company in which they have a 40.0% interest?',

            'What is the Revenue in 2021?',
            'What is the Earnings Before Income Taxes in 2020?',
            'What is the Net Earnings Including Noncontrolling Interests in 2019?',
        ]

@st.cache(
    persist=True,
    allow_output_mutation=True, show_spinner=False,
)
def get_models():
    processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")
    model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    return processor, model

def preprocess_outputs(outputs):
    l = []
    for o in outputs:
        seq = o.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
        seq = re.sub(r"<.*?>", "", seq, count=1).strip()  # remove first task start token
        l.append(processor.token2json(seq))
    return l

st.title('Document Visual Question Answering')
with st.spinner('Load models...'):
    processor, model = get_models()

# print(f'Inter-op threads {torch.get_num_interop_threads()}, threads {torch.get_num_threads()}')
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(8)

pdf_file = st.file_uploader("Choose .pdf file", type='pdf')


if pdf_file:
    bytes_data = pdf_file.getvalue()
    # doc = fitz.open('test_pages/ADM_10K - Highlighted.pdf')  # open document
    doc = fitz.open('pdf', bytes_data)  # open document

    page_num = st.number_input(f'Page Number(from 1 to {len(doc)})', min_value=1, max_value=len(doc), value=1)

    if page_num != st.session_state.page_num:
        st.session_state.page_num = page_num
        st.session_state.block_saved_queries = True

    pix = doc[page_num-1].get_pixmap()  # render page to an image
    # pix.save(f'test_pages/ADM_10K_page{page_num+1}.png' )  # store image as a PNG

    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

    st.image(img, f'Page {page_num} from {len(doc)}')
    with st.spinner('Process image...'):
        pixel_values = processor(img, return_tensors="pt").pixel_values

    text_str = ''
    with st.expander('Saved queries'):
        radio_placeholder = st.empty()
        choice = radio_placeholder.radio('Queries', options=saved_queries, label_visibility='collapsed')
        # if st.button('Use selected query'):
        #     text_str = choice
        # else:
        #     st.stop()

    # if st.session_state.block_saved_queries:
    #     choice = None


    # if choice and use_saved_queries:
    #     input_text = st.text_area('Visual query:', value=choice, height=5, placeholder='Enter your query here')
    # else:
    #     input_text = st.text_area('Visual query:', height=5, placeholder='Enter your query here')
    #     st.session_state.block_saved_queries = False

    input_text = st.text_area('Visual query:', value=choice,  placeholder='Enter your query here')

    if st.button('Query'):
        input_text = input_text.strip()

        # if input_text not in saved_queries:
        #     print(f'add to saved queries: {input_text}')
        #     saved_queries.append(input_text)
        #     choice = radio_placeholder.radio('Queries', options=saved_queries, label_visibility='collapsed')

        prompt_text =f"<s_docvqa><s_question>{input_text}</s_question><s_answer>"

        with st.spinner('Process query...'):
            decoder_input_ids = processor.tokenizer(prompt_text, add_special_tokens=False, padding=True, return_tensors="pt")["input_ids"]

            outputs = model.generate(pixel_values.to(device),
                                           decoder_input_ids=decoder_input_ids.to(device),
                                           max_length=model.decoder.config.max_position_embeddings,
                                           early_stopping=True,
                                           pad_token_id=processor.tokenizer.pad_token_id,
                                           eos_token_id=processor.tokenizer.eos_token_id,
                                           use_cache=True,
                                           num_beams=1,
                                           bad_words_ids=[[processor.tokenizer.unk_token_id]],
                                           return_dict_in_generate=True,
                                           output_scores=True)

            co = preprocess_outputs(processor.batch_decode(outputs.sequences))
            # st.json(co, expanded=True)
            # st.write(co[0]['answer'])
            import pandas as pd
            df = pd.DataFrame(
                co)

            st.table(df)