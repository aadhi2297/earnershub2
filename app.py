import streamlit as st
import pandas as pd
import numpy as np
import os
import random
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from PIL import Image

# Directories
DATA_DIR = 'data/raw/'
QR_DIR = 'assets/qrs/'
LOGO_PATH = 'assets/logo.png'

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(QR_DIR, exist_ok=True)

# Files
REVIEW_FILE = os.path.join(DATA_DIR, 'review_data.csv')
EARNING_FILE = os.path.join(DATA_DIR, 'earning_sources.csv')
REPORTED_FILE = os.path.join(DATA_DIR, 'reported_sources.csv')
REMOVED_FILE = os.path.join(DATA_DIR, 'remove_sources.csv')
WORK_FILE = os.path.join(DATA_DIR, 'work_market_place.csv')

# App config
st.set_page_config(page_title="EarnersHub", page_icon="💸", layout="wide")

# Helper Functions
def load_logo():
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=120)

def show_footer():
    st.markdown("<hr style='margin-top: 30px;'/>", unsafe_allow_html=True)
    st.caption("💸 EarnersHub | Empowering community earnings")

def validate_phone(phone):
    return phone.isdigit() and len(phone) == 10

# Sentiment model
def train_sentiment_model():
    if not os.path.exists(REVIEW_FILE):
        pd.DataFrame(columns=['Date', 'Review', 'Sentiment', 'Source']).to_csv(REVIEW_FILE, index=False)
    df = pd.read_csv(REVIEW_FILE)
    if df.empty:
        return None, None
    X, y = df['Review'], df['Sentiment']
    vect = TfidfVectorizer(ngram_range=(1, 2))
    X_vect = vect.fit_transform(X)
    model = LogisticRegression()
    model.fit(X_vect, y)
    return model, vect

model, vect = train_sentiment_model()

# Sidebar
with st.sidebar:
    load_logo()
    st.title("💸 EarnersHub")
    menu = st.radio("Navigate", ["App Reviews", "Earning Resources", "Removed Resources", "Work Marketplace", "About"])

# App Reviews Section
if menu == "App Reviews":
    load_logo()
    st.markdown("<h2 style='text-align:center;'>📝 App Reviews</h2>", unsafe_allow_html=True)
    st.write("Leave your feedback and see how EarnersHub is doing!")

    with st.form("add_review"):
        review_text = st.text_input("✏️ Your Review")
        submit = st.form_submit_button("Submit Review")
        if submit and review_text:
            sentiment = model.predict(vect.transform([review_text]))[0] if model else 'Positive'
            emoji = '😊' if sentiment == 'Positive' else '😞'
            new_review = pd.DataFrame([[datetime.now().strftime('%Y-%m-%d'), review_text, sentiment, "EarnersHub"]],
                                      columns=['Date', 'Review', 'Sentiment', 'Source'])
            new_review.to_csv(REVIEW_FILE, mode='a', header=False, index=False)
            st.success(f"Review submitted! Sentiment detected: {emoji} **{sentiment}**")
            for _ in range(3):
                st.write("🎈")
            st.balloons()

    df = pd.read_csv(REVIEW_FILE)
    if not df.empty:
        sentiment_counts = df['Sentiment'].value_counts()
        fig, ax = plt.subplots(figsize=(2.5, 2.5))
        wedges, texts, autotexts = ax.pie(
            sentiment_counts,
            labels=[f"{label} ({count})" for label, count in sentiment_counts.items()],
            autopct='%1.1f%%',
            textprops={'fontsize': 8}
        )
        ax.axis('equal')
        col_center = st.columns([2, 1, 2])[1]
        with col_center:
            st.pyplot(fig)
        st.caption(f"📊 Total Reviews: {len(df)}")
    else:
        st.info("No reviews yet.")

    show_footer()
# Earning Resources Section
if menu == "Earning Resources":
    load_logo()
    st.markdown("<h2 style='text-align:center;'>📡 Earning Resources</h2>", unsafe_allow_html=True)

    for file, cols in [(EARNING_FILE, ['Date', 'Name', 'Type', 'Link', 'Submitted_By', 'Trust_Score', 'Image']),
                       (REPORTED_FILE, ['Name', 'Reports'])]:
        if not os.path.exists(file):
            pd.DataFrame(columns=cols).to_csv(file, index=False)

    with st.form("add_source"):
        date = datetime.now().strftime('%Y-%m-%d')
        name = st.text_input("Resource Name")
        type_ = st.selectbox("Type", ["YouTube", "Telegram", "App", "Website", "Other"])
        link = st.text_input("Resource Link")
        submitted_by = st.text_input("Submitted By")
        trust = st.number_input("Trust Score (0-100)", 0, 100)
        image_file = st.file_uploader("Upload Resource Image (optional)")
        add_btn = st.form_submit_button("Add Resource")

        if add_btn and all([name, link, submitted_by]):
            img_path = ""
            if image_file:
                img_path = os.path.join(QR_DIR, f"{name}_{datetime.now().strftime('%Y%m%d%H%M%S')}.png")
                with open(img_path, 'wb') as f:
                    f.write(image_file.read())
            new_source = pd.DataFrame([[date, name, type_, link, submitted_by, trust, img_path]],
                                      columns=['Date', 'Name', 'Type', 'Link', 'Submitted_By', 'Trust_Score', 'Image'])
            new_source.to_csv(EARNING_FILE, mode='a', header=False, index=False)
            st.success("✅ Resource added!")

    sources_df = pd.read_csv(EARNING_FILE)
    reported_df = pd.read_csv(REPORTED_FILE)

    def report_resource(resource_name):
        if resource_name in reported_df['Name'].values:
            reported_df.loc[reported_df['Name'] == resource_name, 'Reports'] += 1
        else:
            new_report = pd.DataFrame([[resource_name, 1]], columns=['Name', 'Reports'])
            reported_df = pd.concat([reported_df, new_report], ignore_index=True)
        reported_df.to_csv(REPORTED_FILE, index=False)
        return reported_df

    updated = False
    for idx, row in sources_df.iterrows():
        st.write(f"**{row['Name']}** ({row['Type']}) — {row['Link']} — Trust Score: {row['Trust_Score']}")
        if isinstance(row['Image'], str) and row['Image'] and os.path.exists(row['Image']):
            st.image(row['Image'], width=120)
        if st.button("🚨 Report", key=f"report_{idx}"):
            reported_df = report_resource(row['Name'])
            updated = True

    removed_df = reported_df[reported_df['Reports'] >= 5]
    if not removed_df.empty:
        removed_log = pd.DataFrame({'source': removed_df['Name'], 'removed_on': datetime.now().strftime('%Y-%m-%d')})
        if not os.path.exists(REMOVED_FILE):
            removed_log.to_csv(REMOVED_FILE, index=False)
        else:
            removed_log.to_csv(REMOVED_FILE, mode='a', header=False, index=False)
        sources_df = sources_df[~sources_df['Name'].isin(removed_df['Name'])]
        sources_df.to_csv(EARNING_FILE, index=False)
        reported_df = reported_df[reported_df['Reports'] < 5]
        reported_df.to_csv(REPORTED_FILE, index=False)

    if not updated:
        st.info("No resources reported yet.")

    show_footer()

# Removed Resources Section
if menu == "Removed Resources":
    load_logo()
    st.markdown("<h2 style='text-align:center;'>🚫 Removed Resources</h2>", unsafe_allow_html=True)

    if os.path.exists(REMOVED_FILE):
        removed_df = pd.read_csv(REMOVED_FILE)
        if removed_df.empty:
            st.info("No resources have been removed yet.")
        else:
            st.dataframe(removed_df)
    else:
        st.info("No resources removed yet.")

    show_footer()
# Work Marketplace Section
if menu == "Work Marketplace":
    load_logo()
    st.markdown("<h2 style='text-align:center;'>🛠️ Work Marketplace</h2>", unsafe_allow_html=True)

    if not os.path.exists(WORK_FILE):
        pd.DataFrame(columns=['Date','Work_Description','Location','Posted_By','Phone','Amount','Accepted_By','UPI_QR','Status','OTP']).to_csv(WORK_FILE, index=False)

    with st.form("post_work"):
        date = datetime.now().strftime('%Y-%m-%d')
        desc = st.text_input("Work Description")
        loc = st.text_input("Location")
        poster = st.text_input("Your Name")
        phone = st.text_input("Phone Number")
        amount = st.text_input("Amount")
        submit_work = st.form_submit_button("Post Work")

        if submit_work and all([desc, loc, poster, phone, amount]):
            if not validate_phone(phone):
                st.error("❌ Invalid Phone Number. Must be 10 digits.")
            else:
                otp = random.randint(1000, 9999)
                new_work = pd.DataFrame([[date, desc, loc, poster, phone, amount, "", "", "Open", otp]],
                                        columns=['Date','Work_Description','Location','Posted_By','Phone','Amount','Accepted_By','UPI_QR','Status','OTP'])
                new_work.to_csv(WORK_FILE, mode='a', header=False, index=False)
                st.success("✅ Work posted successfully.")

    jobs_df = pd.read_csv(WORK_FILE)
    filter_loc = st.text_input("🔍 Filter works by Location")
    if filter_loc:
        jobs_df = jobs_df[jobs_df['Location'].str.contains(filter_loc, case=False, na=False)]

    st.subheader("📋 Open Works")
    st.dataframe(jobs_df[jobs_df['Status'] == 'Open'][['Date','Work_Description','Location','Posted_By','Amount','Status']])

    show_footer()
    # Accept Work Request
    with st.form("accept_work"):
        st.markdown("### 🤝 Accept a Work")
        accept_job = st.text_input("Work Description to Accept")
        accepter_name = st.text_input("Your Name (Accepter)")
        upi_qr = st.file_uploader("Upload your UPI QR (optional)")
        accept_submit = st.form_submit_button("Accept Work")

        if accept_submit and accept_job and accepter_name:
            idx = jobs_df[(jobs_df['Work_Description'] == accept_job) & (jobs_df['Status'] == 'Open')].index
            if not idx.empty:
                upi_path = ""
                if upi_qr:
                    upi_path = os.path.join(QR_DIR, f"{accepter_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}.png")
                    with open(upi_path, 'wb') as f:
                        f.write(upi_qr.read())
                jobs_df.loc[idx, ['Accepted_By', 'UPI_QR', 'Status']] = [accepter_name, upi_path, 'Accepted']
                jobs_df.to_csv(WORK_FILE, index=False)
                st.success(f"✅ Work '{accept_job}' accepted by {accepter_name}!")
                otp_value = jobs_df.loc[idx[0], 'OTP']
                poster_phone = jobs_df.loc[idx[0], 'Phone']
                st.info(f"📱 OTP sent to poster ({poster_phone}). (Simulated: {otp_value})")
            else:
                st.error("❌ Work not found or already accepted.")

    # Delete Work (by poster before acceptance)
    with st.form("delete_work"):
        st.markdown("### ❌ Delete a Work (Before Acceptance)")
        del_job = st.text_input("Work Description to Delete")
        del_poster = st.text_input("Your Name (Poster)")
        del_submit = st.form_submit_button("Delete Work")

        if del_submit and del_job and del_poster:
            idx = jobs_df[(jobs_df['Work_Description'] == del_job) & 
                          (jobs_df['Posted_By'] == del_poster) & 
                          (jobs_df['Status'] == 'Open')].index
            if not idx.empty:
                jobs_df.drop(idx, inplace=True)
                jobs_df.to_csv(WORK_FILE, index=False)
                st.success(f"✅ Work '{del_job}' deleted successfully.")
            else:
                st.error("❌ No matching open work found for deletion.")

    # Complete Work (by accepter using OTP)
    with st.form("complete_work"):
        st.markdown("### ✅ Mark a Work as Completed")
        comp_job = st.text_input("Work Description to Mark Completed")
        accepter_name = st.text_input("Your Name (Accepter)")
        entered_otp = st.text_input("Enter 4-digit OTP", type="password")
        comp_submit = st.form_submit_button("Mark Completed")

        if comp_submit and comp_job and accepter_name and entered_otp:
            idx = jobs_df[(jobs_df['Work_Description'] == comp_job) & 
                          (jobs_df['Accepted_By'] == accepter_name) & 
                          (jobs_df['Status'] == 'Accepted')].index
            if not idx.empty:
                correct_otp = str(jobs_df.loc[idx[0], 'OTP'])
                if entered_otp == correct_otp:
                    jobs_df.drop(idx, inplace=True)
                    jobs_df.to_csv(WORK_FILE, index=False)
                    st.success(f"✅ Work '{comp_job}' marked completed and removed.")
                else:
                    st.error("❌ Incorrect OTP entered.")
            else:
                st.error("❌ No such accepted work found for you.")

    show_footer()
# About Section
if menu == "About":
    load_logo()
    st.markdown("<h2 style='text-align:center;'>ℹ️ About EarnersHub</h2>", unsafe_allow_html=True)

    st.markdown("""
    **EarnersHub** is a community-powered earning discovery and work marketplace platform.

    - 📡 Share trusted **YouTube**, **Telegram**, **App**, or **Website** resources.
    - 📝 Submit reviews and track feedback for the platform itself.
    - 🛠️ Find or post small local/online jobs.
    - 🚨 Community moderation: report untrustworthy resources for automatic removal.

    ### 📊 Highlights:
    - Real-time review sentiment analysis using **Logistic Regression**.
    - Dynamic trust scoring for earning sources.
    - Secure job acceptance with **OTP-based completion confirmation**.
    - Optional image/QR uploads for resources and job payments.

    ---
    👨‍💻 Developed with ❤️ by **G. Adi Shankar**  
    📅 Built using Python, Streamlit, Scikit-learn, and Pandas.
    """)

    show_footer()

