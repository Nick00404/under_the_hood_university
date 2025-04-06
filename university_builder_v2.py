import streamlit as st
import os
import json

# --- CONFIGS ---
empty_notebook = {
    "cells": [],
    "metadata": {},
    "nbformat": 4,
    "nbformat_minor": 2
}

st.set_page_config(page_title="University Builder v2.0", layout="wide")
st.title("📚 University Folder & Notebook Generator")

# --- SESSION STATE ---
if "curriculum" not in st.session_state:
    st.session_state.curriculum = {}

# --- FUNCTIONS ---
def add_module(name):
    if name and name not in st.session_state.curriculum:
        st.session_state.curriculum[name] = {"files": [], "subdirs": {}}
        st.success(f"Module '{name}' added.")

def add_submodule(module, name):
    if name and name not in st.session_state.curriculum[module]["subdirs"]:
        st.session_state.curriculum[module]["subdirs"][name] = {"files": [], "subdirs": {}}
        st.success(f"Submodule '{name}' added under '{module}'.")

def add_file(module, submodule, file_name):
    if submodule:
        target = st.session_state.curriculum[module]["subdirs"][submodule]
    else:
        target = st.session_state.curriculum[module]
    if file_name and file_name not in target["files"]:
        target["files"].append(file_name)
        st.success(f"File '{file_name}' added under module '{module}'" + (f" -> '{submodule}'" if submodule else ""))

def create_structure(base_path, structure):
    for folder, content in structure.items():
        path = os.path.join(base_path, folder)
        os.makedirs(path, exist_ok=True)
        # Create files in the current folder
        for file in content.get("files", []):
            file_path = os.path.join(path, file)
            if file.endswith(".ipynb"):
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(empty_notebook, f, indent=2)
            else:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(f"# {file.replace('.md','').title()}\n")
        # Recursive call for subdirectories
        create_structure(path, content.get("subdirs", {}))

# --- UI FORM ---
project_name = st.text_input("🧱 Project Folder Name", value="university_v2")

with st.expander("➕ Add Modules & Submodules", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        new_module = st.text_input("New Module Name", key="new_module")
        if st.button("Add Module", key="btn_module"):
            add_module(new_module)
    with col2:
        if st.session_state.curriculum:
            module_select = st.selectbox("Select Module", list(st.session_state.curriculum.keys()), key="module_select")
            new_submodule = st.text_input("New Submodule Name", key="new_submodule")
            if st.button("Add Submodule", key="btn_submodule"):
                add_submodule(module_select, new_submodule)

st.markdown("---")
# --- FILE ADDING ---
if st.session_state.curriculum:
    st.subheader("📁 Add Files to Modules/Submodules")
    mod = st.selectbox("Module", list(st.session_state.curriculum.keys()), key="file_module")
    subs = list(st.session_state.curriculum[mod]["subdirs"].keys())
    submod = st.selectbox("Submodule (optional)", [""] + subs, key="file_submodule")
    new_file = st.text_input("File Name (e.g. 01_intro.ipynb or README.md)", key="new_file")
    if st.button("Add File", key="btn_file"):
        add_file(mod, submod if submod != "" else None, new_file)

st.markdown("---")
# --- DISPLAY STRUCTURE ---
st.subheader("📂 Current Structure Preview")
st.json(st.session_state.curriculum)

# --- EXPORT & LOAD ---
col1, col2 = st.columns(2)
with col1:
    curriculum_json = json.dumps(st.session_state.curriculum, indent=2)
    st.download_button("📥 Export Curriculum JSON", data=curriculum_json, file_name="curriculum.json")
with col2:
    uploaded = st.file_uploader("📤 Load Curriculum", type="json")
    if uploaded is not None:
        loaded_struct = json.load(uploaded)
        st.session_state.curriculum = loaded_struct
        st.success("Curriculum loaded!")

st.markdown("---")
# --- GENERATE FOLDER ---
if st.button("🚀 Generate Folder Structure"):
    base_path = os.path.join(os.getcwd(), project_name)
    os.makedirs(base_path, exist_ok=True)
    create_structure(base_path, st.session_state.curriculum)
    st.success(f"Generated folder structure under '{project_name}'")
