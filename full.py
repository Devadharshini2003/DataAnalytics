import os
import time
import pandas as pd
import streamlit as st
from streamlit_chat import message
from dotenv import load_dotenv
import google.generativeai as genai
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize Streamlit session state for navigation
if "page" not in st.session_state:
    st.session_state.page = "opening"

# Function to change pages
def navigate_to(page):
    st.session_state.page = page
    st.rerun()

# Function to load API key
def load_api_key():
    load_dotenv()
    API_KEY = os.getenv("GEMINI_API_KEY")
    genai.configure(api_key=API_KEY)

# Function to upload file to Gemini
def upload_to_gemini(path, mime_type):
    """Uploads the given file to Gemini and returns the file object."""
    file = genai.upload_file(path, mime_type=mime_type)
    return file

# Function to check if file processing is complete
def wait_for_files_active(files):
    """Waits for files to be ready for use after uploading."""
    st.info("Processing your file...")
    for file in files:
        while file.state.name == "PROCESSING":
            time.sleep(5)
            file = genai.get_file(file.name)
        if file.state.name != "ACTIVE":
            raise Exception(f"File {file.name} failed to process.")
    st.success("Your file is ready!")

# Function to clean data (handle missing values and remove duplicates)
def clean_data(df):
    df = df.drop_duplicates()  # Remove duplicate rows
    df = df.ffill().bfill()  # Fill missing values using nearby values
    return df

# Opening Screen
if st.session_state.page == "opening":
    st.title("📊 AI Data Analysis Toolkit")
    st.subheader("Unlock Insights with AI-Powered Data Visualization")
    
    if st.button("Get Started"):
        navigate_to("home")

# Home Screen (Options Page)
elif st.session_state.page == "home":
    st.title("Home - Select an Analysis Option")

    if st.button("📂 PDF & CSV Analysis"):
        navigate_to("analysis")

    if st.button("📊 Select Your Chart & Axes"):
        navigate_to("visualize_axes")

    if st.button("💡 Summary & Chart"):
        navigate_to("summary_chart")

# Visualization Page (Data Upload & Chart Selection)
elif st.session_state.page == "visualize_axes":
    st.title("📊 Data Upload and Visualization")
    
    uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx"])

    if uploaded_file is not None:
        try:
            # Load Data
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:  # Excel file
                df = pd.read_excel(uploaded_file)

            # Clean Data (Handle Missing Values & Remove Duplicates)
            cleaned_df = clean_data(df)

            # Allow User to Modify Data
            st.subheader("Modify Your Data (Optional)")
            edited_df = st.data_editor(cleaned_df)

            # Clean Again After Modification
            final_df = clean_data(edited_df)

            # Show Final Cleaned Data
            st.subheader("Final Processed Data")
            st.dataframe(final_df)

            # Select Columns for Visualization
            columns = list(final_df.columns)

            # Chart Type Selection
            chart_type = st.selectbox("Select Chart Type", ["Scatter", "Line", "Bar", "Pie"])

            if chart_type != "Pie":
                x_axis = st.selectbox("Select X-axis", columns)
                y_axis = st.selectbox("Select Y-axis", columns)
                color = st.selectbox("Color by (Optional)", ["None"] + columns)

            else:  # Pie Chart
                names = st.selectbox("Select Category Column", columns)
                values = st.selectbox("Select Values Column", columns)

                # Ensure Values Column is Numeric
                if not pd.api.types.is_numeric_dtype(final_df[values]):
                    st.error("The 'Values' column must be numeric for a Pie chart.")
                    st.stop()

            # Generate Chart
            if st.button("Generate Chart"):
                if chart_type == "Scatter":
                    fig = px.scatter(final_df, x=x_axis, y=y_axis, color=None if color == "None" else color)
                elif chart_type == "Line":
                    fig = px.line(final_df, x=x_axis, y=y_axis, color=None if color == "None" else color)
                elif chart_type == "Bar":
                    fig = px.bar(final_df, x=x_axis, y=y_axis, color=None if color == "None" else color)
                elif chart_type == "Pie":
                    fig = px.pie(final_df, names=names, values=values)

                st.plotly_chart(fig)

        except Exception as e:
            st.error(f"An error occurred: {e}")

    if st.button("⬅️ Back to Home"):
        navigate_to("home")

# PDF & CSV Analysis Selection Page
elif st.session_state.page == "analysis":
    st.title("📂 PDF & CSV Analysis")
    
    if st.button("📄 PDF Analysis"):
        navigate_to("pdf_analysis")

    if st.button("📊 CSV Analysis"):
        navigate_to("csv_analysis")

    if st.button("⬅️ Back to Home"):
        navigate_to("home")

# PDF Analysis Page (With AI Chatbot)
elif st.session_state.page == "pdf_analysis":
    st.title("📄 PDF Chat Bot - Ask Questions from Your PDF")
    
    load_api_key()

    uploaded_file = st.file_uploader("Upload your PDF file", type=["pdf"])
    
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    if uploaded_file:
        with open("temp_file.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        files = [upload_to_gemini("temp_file.pdf", "application/pdf")]
        wait_for_files_active(files)

        generation_config = {
            "temperature": 0.8,
            "top_p": 0.95,
            "top_k": 50,
            "max_output_tokens": 1024,
            "response_mime_type": "text/plain",
        }
        model = genai.GenerativeModel(model_name="gemini-1.5-flash", generation_config=generation_config)

        chat_session = model.start_chat(
            history=[{
                "role": "user",
                "parts": [
                    files[0],
                    "Provide answers based on the events in this PDF file."
                ]
            }]
        )

        def ask_question(question):
            """Sends a question and returns AI response."""
            response = chat_session.send_message(question)
            return response.text if response.text.strip() else "No relevant information found in the PDF."

        user_question = st.chat_input("Ask a question from your PDF:")

        if user_question:
            st.session_state.messages.append({"content": user_question, "is_user": True})
            with st.spinner("Thinking..."):
                response_text = ask_question(user_question)
            st.session_state.messages.append({"content": response_text, "is_user": False})

        for i, msg in enumerate(st.session_state.messages):
            message(msg["content"], is_user=msg["is_user"], key=f"{i}")

    if st.button("⬅️ Back to PDF & CSV Analysis"):
        navigate_to("analysis")

# CSV Analysis Page (With AI Chatbot)
elif st.session_state.page == "csv_analysis":
    st.title("📊 CSV Data Analysis - Ask AI Questions about Your Data")
    
    load_api_key()

    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    
    if "csv_messages" not in st.session_state:
        st.session_state["csv_messages"] = []

    if uploaded_file:
        with open("temp_file.csv", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        files = [upload_to_gemini("temp_file.csv", "text/csv")]
        wait_for_files_active(files)

        generation_config = {
            "temperature": 0.8,
            "top_p": 0.95,
            "top_k": 50,
            "max_output_tokens": 1024,
            "response_mime_type": "text/plain",
        }
        model = genai.GenerativeModel(model_name="gemini-1.5-flash", generation_config=generation_config)

        chat_session = model.start_chat(
            history=[{
                "role": "user",
                "parts": [
                    files[0],
                    "Analyze this CSV dataset and provide insights based on user questions."
                ]
            }]
        )

        def ask_csv_question(question):
            """Sends a question and returns AI response."""
            response = chat_session.send_message(question)
            return response.text if response.text.strip() else "No relevant information found in the CSV."

        user_question = st.chat_input("Ask a question from your CSV:")

        if user_question:
            st.session_state.csv_messages.append({"content": user_question, "is_user": True})
            with st.spinner("Thinking..."):
                response_text = ask_csv_question(user_question)
            st.session_state.csv_messages.append({"content": response_text, "is_user": False})

        for i, msg in enumerate(st.session_state.csv_messages):
            message(msg["content"], is_user=msg["is_user"], key=f"csv_{i}")

    if st.button("⬅️ Back to PDF & CSV Analysis"):
        navigate_to("analysis")

# Summary & Chart Page (Third Module)
elif st.session_state.page == "summary_chart":
    st.title("💡 AI-Powered Data Upload, Editing, and Insights")
    
    load_api_key()

    uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx"])

    if uploaded_file:
        try:
            # Load Data
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            # Clean Data
            cleaned_df = clean_data(df)

            # Allow User to Modify Data
            st.subheader("Modify Your Data (Optional)")
            edited_df = st.data_editor(cleaned_df)

            # Clean Again After Modification
            final_df = clean_data(edited_df)

            # Show Final Cleaned Data
            st.subheader("Final Processed Data")
            st.dataframe(final_df)

            # Save uploaded file locally for Gemini processing
            temp_file_path = "uploaded_data.csv"
            final_df.to_csv(temp_file_path, index=False)

            # Upload to Gemini
            files = [upload_to_gemini(temp_file_path, "text/csv")]


            # Wait for files to become active
            wait_for_files_active(files)

            # Initialize AI chat
            generation_config = {
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 50,
                "max_output_tokens": 8192,
                "response_mime_type": "text/plain",
            }
            model = genai.GenerativeModel("gemini-1.5-pro", generation_config=generation_config)
            chat_session = model.start_chat(history=[
                {
                    "role": "user",
                    "parts": [
                        files[0],
                        "You are an AI that analyzes CSV data and provides insights and charts based on user queries."
                        "Ensure the response contains both a textual analysis and a recommendation of sales that boosts their growth.the user may upload any csv file containing different types of columns and rows , so analyse the file and provide them with the correct data analysis chart and also provide them with the summary and the advancements they can make"
                        "one important thing is you should visually provide the chart in the UI and the user needs the charts for every query they asks so give the relavant one by anaysing the question If no chart can be generated for the query , understand the query and provide chart for the meaning of the query"
                        "Dont provide any code snippets on the ui"
                    ],  
                },
            ])

            # User query input
            user_query = st.text_input("Ask any question about your data:")

            if user_query:
                # AI Response
                formatted_query = f"Analyze the uploaded CSV and answer: {user_query}. Also, suggest or generate a chart that best represents the insights."
                response = chat_session.send_message(formatted_query)

                if response.text:
                    st.write("### AI Response:")
                    st.write(response.text)

                # Generate dynamic chart based on the user's query
                def generate_dynamic_chart(df, query):
                    analysis_query = f"Based on the query '{query}', recommend a chart type and columns to use from the uploaded data."
                    chart_response = chat_session.send_message(analysis_query)

                    st.write("### Chart Recommendation from AI:")
                    st.write(chart_response.text)

                    try:
                        if "bar chart" in chart_response.text.lower():
                            columns = [col for col in df.columns if col in chart_response.text]
                            if len(columns) >= 2:
                                grouped_data = df.groupby(columns[0])[columns[1]].sum().reset_index()
                                fig, ax = plt.subplots(figsize=(8, 5))
                                sns.barplot(data=grouped_data, x=columns[0], y=columns[1], ax=ax)
                                ax.set_title(f"Bar Chart of {columns[1]} by {columns[0]}")
                                return fig
                            else:
                                st.warning("AI couldn't determine enough columns for chart generation.")
                        elif "line chart" in chart_response.text.lower():
                            columns = [col for col in df.columns if col in chart_response.text]
                            if len(columns) >= 2:
                                fig, ax = plt.subplots(figsize=(8, 5))
                                df.plot(x=columns[0], y=columns[1], kind="line", ax=ax)
                                ax.set_title(f"Line Chart of {columns[1]} over {columns[0]}")
                                return fig
                        else:
                            st.warning("AI did not provide a recognizable chart type.")
                    except Exception as e:
                        st.error(f"Error generating chart: {e}")
                        return None

                # Display generated chart
                chart = generate_dynamic_chart(final_df, user_query)
                if chart:
                    st.write("### Generated Chart:")
                    st.pyplot(chart)

        except Exception as e:
            st.error(f"An error occurred: {e}")

    if st.button("⬅️ Back to Home"):
        navigate_to("home")
