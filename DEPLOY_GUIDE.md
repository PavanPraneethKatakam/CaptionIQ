# Deployment Guide for CaptionIQ

This document explains how to deploy your Image Captioning application to the public.

## 1. Hugging Face Spaces (Streamlit SDK)

Hugging Face Spaces allows you to quickly host Machine Learning demos. I have prepared an automated script to deploy it.

### Automatic Method (Recommended)
You can deploy directly from your local terminal using the script provided:
```bash
python3 deploy_hf.py --token YOUR_HUGGING_FACE_TOKEN --repo-name CaptionIQ
```
*Note: Make sure your token has **Write** permissions. You can generate one at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).*

The script will:
1. Automatically create a Streamlit Space named `CaptionIQ` on your Hugging Face account.
2. Filter out unnecessary large files (like dataset images in `data/Images`).
3. Upload your models and code.
4. Set up Git LFS (Large File Storage) for your `.h5` model files automatically.

## 2. Streamlit Community Cloud

Streamlit provides a free community cloud that integrates directly with GitHub. If you prefer to deploy there:

1. **Push your code to GitHub**:
   - Create a GitHub repository for your project.
   - Commit and push your code. Because the `.h5` models are large, ensure you have Git LFS installed (`git lfs install`) before pushing, or exclude them and load models from an external URL. I have added `.gitattributes` to track `.h5` files with LFS.

2. **Deploy via Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io/).
   - Click "New app".
   - Select your GitHub repository, choose the main branch, and specify `app.py` as your Main file path.
   - Click **Deploy**.

> **Note on Free Tiers**: Streamlit Community Cloud has a 1GB memory limit which might be tight when loading two large VGG models simultaneously. Hugging Face Spaces (Free Tier) typically offers 16GB of RAM, making it a better fit for this project!
