import streamlit as st
import pandas as pd
import numpy as np
import os
import re
import time
import random
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer  # ✅ add this
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from PIL import Image
import matplotlib.pyplot as plt


# -----------------------------
# Configuration & Paths
# -----------------------------
DATA_DIR = 'data/raw/'
QR_DIR = 'assets/qrs/'
ASSETS_DIR = 'assets/'
LOGO_PATH = os.path.join(ASSETS_DIR, 'logo.png')

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(QR_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)

REVIEW_FILE = os.path.join(DATA_DIR, 'review_data.csv')
EARNING_FILE = os.path.join(DATA_DIR, 'earning_sources.csv')
REPORTED_FILE = os.path.join(DATA_DIR, 'reported_sources.csv')
REMOVED_FILE = os.path.join(DATA_DIR, 'removed_sources.csv')
WORK_FILE = os.path.join(DATA_DIR, 'work_market_place.csv')

st.set_page_config(page_title="EarnersHub", page_icon="💸", layout="wide")

# -----------------------------
# Utility functions
# -----------------------------

def load_logo(width=120):
    if os.path.exists(LOGO_PATH):
        try:
            img = Image.open(LOGO_PATH)
            st.image(img, width=width)
        except Exception:
            pass


def show_footer():
    st.markdown("<hr style='margin-top: 30px;'/>", unsafe_allow_html=True)
    st.caption("💸 EarnersHub | Empowering community earnings")


def validate_phone(phone: str) -> bool:
    return str(phone).isdigit() and len(str(phone)) == 10


# Safe CSV read to handle empty files
def safe_read_csv(path, dtype=None):
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, dtype=dtype)
        return df
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


# -----------------------------
# Sentiment Model (simple retrainable wrapper)
# -----------------------------
class SentimentModel:
    def __init__(self, review_file=REVIEW_FILE):
        self.review_file = review_file
        self.vect = None
        self.model = None
        self._train()

    def _ensure_file(self):
        if not os.path.exists(self.review_file):
            pd.DataFrame(columns=['Date', 'Review', 'Sentiment', 'Source']).to_csv(self.review_file, index=False)

    def _train(self):
        self._ensure_file()
        df = safe_read_csv(self.review_file)
        if df.empty or 'Sentiment' not in df.columns or df['Sentiment'].nunique() < 2:
            # Not enough labeled data: keep model None and use rule-based fallback
            self.model = None
            self.vect = None
            return
        X = df['Review'].astype(str).values
        y = df['Sentiment'].astype(str).values
        self.vect = TfidfVectorizer(ngram_range=(1,2), max_features=5000)
        Xv = self.vect.fit_transform(X)
        self.model = LogisticRegression(max_iter=1000)
        self.model.fit(Xv, y)

    def predict(self, texts):
        texts = [str(t) for t in (texts if isinstance(texts, (list, np.ndarray)) else [texts])]
        if self.model is None or self.vect is None:
            # fallback - very simple rule based
            results = []
            positive_words = set(['good','great','love','best','amazing','helpful','useful','awesome','trust'])
            for t in texts:
                tokens = set([w.lower().strip(".,!?") for w in t.split()])
                results.append('Positive' if len(tokens & positive_words) >= 1 else 'Negative')
            return results if len(results) > 1 else results[0]
        try:
            Xv = self.vect.transform(texts)
            preds = self.model.predict(Xv)
            return preds if len(preds)>1 else preds[0]
        except NotFittedError:
            return self.predict(texts)

    def retrain(self):
        self._train()


sentiment_model = SentimentModel()

# -----------------------------
# Ensure files exist with correct headers
# -----------------------------
if not os.path.exists(REVIEW_FILE):
    pd.DataFrame(columns=['Date', 'Review', 'Sentiment', 'Source']).to_csv(REVIEW_FILE, index=False)
if not os.path.exists(EARNING_FILE):
    pd.DataFrame(columns=['Date', 'Name', 'Type', 'Link', 'Submitted_By', 'Trust_Score', 'Image']).to_csv(EARNING_FILE, index=False)
if not os.path.exists(REPORTED_FILE):
    pd.DataFrame(columns=['Name', 'Reports']).to_csv(REPORTED_FILE, index=False)
if not os.path.exists(REMOVED_FILE):
    pd.DataFrame(columns=['source', 'removed_on']).to_csv(REMOVED_FILE, index=False)
if not os.path.exists(WORK_FILE):
    pd.DataFrame(columns=['Date','Work_Description','Location','Posted_By','Phone','Amount','Accepted_By','UPI_QR','Status','OTP']).to_csv(WORK_FILE, index=False)

# -----------------------------
# Sidebar & Navigation
# -----------------------------
with st.sidebar:
    load_logo(120)
    st.title("💸 EarnersHub")
    menu = st.radio("Navigate", ["App Reviews", "Earning Resources", "Removed Resources", "Work Marketplace", "About"]) 

# -----------------------------
# App Reviews
# -----------------------------
if menu == "App Reviews":
    st.markdown("<div style='display:flex;align-items:center;justify-content:space-between;'>" +
                "<h2>📝 App Reviews</h2>" +
                "</div>", unsafe_allow_html=True)

    st.write("Share feedback and help the community — sentiment is shown immediately after submission.")

    with st.form("add_review"):
        review_text = st.text_area("✏️ Your Review", height=120)
        source = st.text_input("Source (optional)")
        submit = st.form_submit_button("Submit Review")
        if submit and review_text.strip():
            sentiment = sentiment_model.predict(review_text)
            date = datetime.now().strftime('%Y-%m-%d')
            new_row = pd.DataFrame([[date, review_text.strip(), sentiment, source or 'EarnersHub']],
                                   columns=['Date', 'Review', 'Sentiment', 'Source'])
            new_row.to_csv(REVIEW_FILE, mode='a', header=False, index=False)
            st.success(f"Review submitted — Sentiment detected: {'😊 Positive' if sentiment=='Positive' else '😞 Negative'}")
            st.balloons()
            # retrain model if there are now labeled examples
            sentiment_model.retrain()

    reviews_df = safe_read_csv(REVIEW_FILE)
    if not reviews_df.empty:
        reviews_df = reviews_df.sort_values(by='Date', ascending=False).head(25).reset_index(drop=True)
        # show small pie chart of sentiment distribution
        # prepare table for display (avoid using pandas Styler because Streamlit has limited Styler support)
        # show small pie chart of sentiment distribution
        # Sentiment Distribution Pie Chart (sharp and clean)
        sentiment_counts = reviews_df['Sentiment'].value_counts()
        labels = [f"{label}" for label in sentiment_counts.index]
        sizes = sentiment_counts.values
        colors = ["#8fd694", "#ff9999"]  # green for positive, red for negative

        fig, ax = plt.subplots(figsize=(2, 2), dpi=120)  # bigger size + higher DPI
        wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        autopct='%1.1f%%',
        startangle=140,
        colors=colors,
        wedgeprops={'linewidth': 0}
)

        for text in texts:
         text.set_fontsize(11)
         text.set_color("black")
        for autotext in autotexts:
         autotext.set_fontsize(10)
         autotext.set_color("white")

        ax.set_title("Sentiment Distribution", fontsize=13, weight="bold")
        st.pyplot(fig, clear_figure=True)


    else:
        st.info("No reviews yet — be the first to submit!")

    show_footer()

# -----------------------------
# Earning Resources
# -----------------------------
if menu == "Earning Resources":
    st.markdown("<h2 style='text-align:center;'>📡 Earning Resources</h2>", unsafe_allow_html=True)

    with st.form("add_source"):
        date = datetime.now().strftime('%Y-%m-%d')
        name = st.text_input("Resource Name")
        type_ = st.selectbox("Type", ["YouTube", "Telegram", "App", "Website", "Other"]) 
        link = st.text_input("Resource Link / Channel")
        submitted_by = st.text_input("Submitted By")
        trust = st.number_input("Trust Score (0-100)", 0, 100, value=50)
        image_file = st.file_uploader("Upload Resource Image (optional)")
        add_btn = st.form_submit_button("Add Resource")

        if add_btn:
            if not (name and link and submitted_by):
                st.error("Please provide Name, Link and Submitted By fields.")
            else:
                img_path = ""
                if image_file is not None:
                    safe_name = ''.join(c for c in name if c.isalnum() or c in [' ', '_', '-']).rstrip()
                    img_path = os.path.join(QR_DIR, f"{safe_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}.png")
                    with open(img_path, 'wb') as f:
                        f.write(image_file.read())
                new_source = pd.DataFrame([[date, name.strip(), type_, link.strip(), submitted_by.strip(), trust, img_path]],
                                          columns=['Date', 'Name', 'Type', 'Link', 'Submitted_By', 'Trust_Score', 'Image'])
                new_source.to_csv(EARNING_FILE, mode='a', header=False, index=False)
                st.success("✅ Resource added!")

    sources_df = safe_read_csv(EARNING_FILE)
    reported_df = safe_read_csv(REPORTED_FILE)

    # Make sure reported_df has correct dtypes
    if reported_df.empty:
        reported_df = pd.DataFrame(columns=['Name','Reports'])

    st.subheader('Browse Resources')
    if sources_df.empty:
        st.info('No resources available yet.')
    else:
        # Show only the most recent 100
        sources_df = sources_df.sort_values(by='Date', ascending=False).head(100).reset_index(drop=True)
        for idx, row in sources_df.iterrows():
            cols = st.columns([0.8, 2, 1, 0.5])
            with cols[0]:
                st.markdown(f"**{row['Name']}**  ")
                st.caption(f"Type: {row['Type']}")
                st.write(row['Link'])
                st.caption(f"Trust Score: {int(row['Trust_Score'])}")
            with cols[1]:
                if isinstance(row.get('Image',''), str) and row['Image'] and os.path.exists(row['Image']):
                    st.image(row['Image'], width=120)
            with cols[2]:
                # Report button
                if st.button('🚨 Report', key=f"report_{idx}"):
                    name = row['Name']
                    if name in reported_df['Name'].values:
                        reported_df.loc[reported_df['Name']==name,'Reports'] += 1
                    else:
                        reported_df = pd.concat([reported_df, pd.DataFrame([[name,1]], columns=['Name','Reports'])], ignore_index=True)
                    reported_df.to_csv(REPORTED_FILE, index=False)
                    st.success(f"Reported {name}. Total reports: {int(reported_df[reported_df['Name']==name]['Reports'].iloc[0])}")
                    # after report, if threshold reached, remove it
                    if int(reported_df[reported_df['Name']==name]['Reports'].iloc[0]) >= 5:
                        removed_log = pd.DataFrame({'source':[name], 'removed_on':[datetime.now().strftime('%Y-%m-%d')]})
                        removed_log.to_csv(REMOVED_FILE, mode='a', header=False, index=False)
                        sources_df = sources_df[sources_df['Name'] != name]
                        sources_df.to_csv(EARNING_FILE, index=False)
                        reported_df = reported_df[reported_df['Reports'] < 5]
                        reported_df.to_csv(REPORTED_FILE, index=False)
                        st.warning(f"{name} has been removed after 5 reports.")
            with cols[3]:
                st.write('')

    show_footer()

# -----------------------------
# Removed Resources
# -----------------------------
if menu == "Removed Resources":
    st.markdown("<h2 style='text-align:center;'>🚫 Removed Resources</h2>", unsafe_allow_html=True)
    removed_df = safe_read_csv(REMOVED_FILE)
    if removed_df.empty:
        st.info('No resources have been removed yet.')
    else:
        st.dataframe(removed_df, use_container_width=True)
    show_footer()

# -----------------------------
# Work Marketplace
# -----------------------------
if menu == "Work Marketplace":
    st.markdown("<h2 style='text-align:center;'>🛠️ Work Marketplace</h2>", unsafe_allow_html=True)

    with st.form('post_work'):
        date = datetime.now().strftime('%Y-%m-%d')
        desc = st.text_input('Work Description')
        loc = st.text_input('Location')
        poster = st.text_input('Your Name')
        phone = st.text_input('Phone Number')
        amount = st.text_input('Amount')
        submit_work = st.form_submit_button('Post Work')
        if submit_work:
            if not all([desc.strip(), loc.strip(), poster.strip(), phone.strip(), amount.strip()]):
                st.error('Please fill all fields.')
            elif not validate_phone(phone):
                st.error('❌ Invalid Phone Number. Must be 10 digits.')
            else:
                otp = random.randint(1000,9999)
                new_work = pd.DataFrame([[date, desc.strip(), loc.strip(), poster.strip(), phone.strip(), amount.strip(), '', '', 'Open', otp]],
                                        columns=['Date','Work_Description','Location','Posted_By','Phone','Amount','Accepted_By','UPI_QR','Status','OTP'])
                new_work.to_csv(WORK_FILE, mode='a', header=False, index=False)
                st.success('✅ Work posted successfully.')

    jobs_df = safe_read_csv(WORK_FILE)
    if jobs_df.empty:
        jobs_df = pd.DataFrame(columns=['Date','Work_Description','Location','Posted_By','Phone','Amount','Accepted_By','UPI_QR','Status','OTP'])

    filter_loc = st.text_input('🔍 Filter works by Location')
    display_jobs = jobs_df.copy()
    if filter_loc.strip():
        display_jobs = display_jobs[display_jobs['Location'].str.contains(filter_loc.strip(), case=False, na=False)]

    st.subheader('📋 Open Works')
    st.dataframe(display_jobs[display_jobs['Status']=='Open'][['Date','Work_Description','Location','Posted_By','Amount','Status']], use_container_width=True)

    # Accept Work
    with st.form('accept_work'):
        st.markdown('### 🤝 Accept a Work')
        accept_job = st.selectbox('Select Work to Accept', options=(jobs_df[jobs_df['Status']=='Open']['Work_Description'].tolist() or ['--none--']))
        accepter_name = st.text_input('Your Name (Accepter)')
        upi_qr = st.file_uploader('Upload your UPI QR (optional)')
        accept_submit = st.form_submit_button('Accept Work')
        if accept_submit:
            if accept_job == '--none--' or not accepter_name.strip():
                st.error('Please choose an open work and enter your name.')
            else:
                idx = jobs_df[(jobs_df['Work_Description']==accept_job) & (jobs_df['Status']=='Open')].index
                if idx.empty:
                    st.error('Work not found or already accepted.')
                else:
                    upi_path = ''
                    if upi_qr is not None:
                        safe_name = ''.join(c for c in accepter_name if c.isalnum() or c in [' ', '_', '-']).rstrip()
                        upi_path = os.path.join(QR_DIR, f"{safe_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}.png")
                        with open(upi_path, 'wb') as f:
                            f.write(upi_qr.read())
                    jobs_df.loc[idx, ['Accepted_By','UPI_QR','Status']] = [accepter_name.strip(), upi_path, 'Accepted']
                    jobs_df.to_csv(WORK_FILE, index=False)
                    st.success(f"✅ Work '{accept_job}' accepted by {accepter_name.strip()}!")
                    otp_value = jobs_df.loc[idx[0], 'OTP']
                    poster_phone = jobs_df.loc[idx[0], 'Phone']
                    st.info(f"📱 OTP sent to poster ({poster_phone}). (Simulated: {otp_value})")

    # Delete Work (before acceptance) - only poster can delete
    with st.form('delete_work'):
        st.markdown('### ❌ Delete a Work (Before Acceptance)')
        del_job = st.selectbox('Select Work to Delete', options=(jobs_df[jobs_df['Status']=='Open']['Work_Description'].tolist() or ['--none--']))
        del_poster = st.text_input('Your Name (Poster)')
        del_submit = st.form_submit_button('Delete Work')
        if del_submit:
            if del_job == '--none--' or not del_poster.strip():
                st.error('Choose a work and enter your name.')
            else:
                idx = jobs_df[(jobs_df['Work_Description']==del_job) & (jobs_df['Posted_By']==del_poster.strip()) & (jobs_df['Status']=='Open')].index
                if idx.empty:
                    st.error('❌ No matching open work found for deletion (only the poster can delete an open work).')
                else:
                    jobs_df.drop(idx, inplace=True)
                    jobs_df.to_csv(WORK_FILE, index=False)
                    st.success(f"✅ Work '{del_job}' deleted successfully.")

    # Complete Work (by accepter using OTP)
    with st.form('complete_work'):
        st.markdown('### ✅ Mark a Work as Completed')
        comp_job = st.selectbox('Select Accepted Work to Complete', options=(jobs_df[jobs_df['Status']=='Accepted']['Work_Description'].tolist() or ['--none--']))
        accepter_name = st.text_input('Your Name (Accepter) for Completion')
        entered_otp = st.text_input('Enter 4-digit OTP', type='password')
        comp_submit = st.form_submit_button('Mark Completed')
        if comp_submit:
            if comp_job == '--none--' or not accepter_name.strip() or not entered_otp.strip():
                st.error('Please select a work, enter your name, and the OTP.')
            else:
                idx = jobs_df[(jobs_df['Work_Description']==comp_job) & (jobs_df['Accepted_By']==accepter_name.strip()) & (jobs_df['Status']=='Accepted')].index
                if idx.empty:
                    st.error('❌ No such accepted work found for you.')
                else:
                    correct_otp = str(jobs_df.loc[idx[0], 'OTP'])
                    if entered_otp.strip() == correct_otp:
                        jobs_df.drop(idx, inplace=True)
                        jobs_df.to_csv(WORK_FILE, index=False)
                        st.success(f"✅ Work '{comp_job}' marked completed and removed.")
                    else:
                        st.error('❌ Incorrect OTP entered.')

    # Show Accepted Works with UPI preview
    st.subheader('Accepted Works')
    accepted = jobs_df[jobs_df['Status']=='Accepted']
    if accepted.empty:
        st.info('No accepted works at the moment.')
    else:
        for _, r in accepted.iterrows():
            cols = st.columns([2,1])
            with cols[0]:
                st.markdown(f"**{r['Work_Description']}** — Accepted by: {r['Accepted_By']} — Amount: {r['Amount']}")
                st.caption(f"Posted by: {r['Posted_By']} | Location: {r['Location']}")
            with cols[1]:
                if isinstance(r.get('UPI_QR',''), str) and r['UPI_QR'] and os.path.exists(r['UPI_QR']):
                    st.image(r['UPI_QR'], width=120)

    show_footer()

# -----------------------------
# About
# -----------------------------
if menu == "About":
    st.markdown("<h2 style='text-align:center;'>ℹ️ About EarnersHub</h2>", unsafe_allow_html=True)
    st.markdown(r"""
    **EarnersHub** is a community-powered earning discovery and work marketplace platform.

    - 📡 Share trusted **YouTube**, **Telegram**, **App**, or **Website** resources.
    - 📝 Submit reviews and track feedback for the platform itself.
    - 🛠️ Find or post small local/online jobs.
    - 🚨 Community moderation: report untrustworthy resources for automatic removal.

    ### 📊 Highlights:
    - Real-time review sentiment analysis using **Logistic Regression** (when there is enough labeled data).
    - Dynamic trust scoring for earning sources.
    - Secure job acceptance with **OTP-based completion confirmation**.
    - Optional image/QR uploads for resources and job payments.

    ---
    👨‍💻 Developed with ❤️ by **G. Adi Shankar**  
    📅 Built using Python, Streamlit, Scikit-learn, and Pandas.
    """)
    show_footer()
