"""
Mocker - AI-Infused Smart Expense, Savings & Investment Manager
===============================================================
B.Tech CSE (AI/ML) Mini Project - 3rd Year

Modules
-------
1. Dashboard            : Key metrics + live countdown to every financial goal.
2. Add Expense          : Live NLP categorisation (TF-IDF + Multinomial Naive Bayes).
3. View & Analyze       : Full CRUD on expenses + category analytics.
4. Money Management     : Emergency Fund tracker + Savings Goals with pie chart & history.
5. Investment Manager   : Holdings, Invested vs Earned vs Spent analysis, Investment goals.
6. AI Financial News    : Synthesised market insight feed.
7. Settings             : Demo data loader, CSV export, database reset.

Storage : SQLite (relational, ACID, accessed through parameterised SQL).
Run     : python -m streamlit run Mocker.py
"""

import os
import sqlite3
from datetime import date, datetime

import pandas as pd
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend: required for a web server context.
import matplotlib.pyplot as plt

import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# =============================================================================
# 1. PAGE CONFIGURATION  (this is what renames the browser tab to "Mocker")
# =============================================================================
st.set_page_config(
    page_title="Mocker",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

DB_FILE = "mocker.db"        # Relational database file (auto-created on first run)
LEGACY_CSV = "expenses.csv"  # Old flat-file storage, imported automatically once

# Matplotlib's default font has no glyph for the rupee sign, so charts use "Rs".
RUPEE = "₹"


def money(value):
    """Format a number as Indian rupees for display in the UI."""
    return f"{RUPEE}{value:,.2f}"


# =============================================================================
# 2. DATABASE LAYER  (plain parameterised SQL - safe against SQL injection)
# =============================================================================
def get_conn():
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def execute(sql, params=()):
    """Run an INSERT / UPDATE / DELETE statement. Returns the new row id."""
    conn = get_conn()
    try:
        cur = conn.execute(sql, params)
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def query_df(sql, params=()):
    """Run a SELECT statement and return the result as a pandas DataFrame."""
    conn = get_conn()
    try:
        return pd.read_sql_query(sql, conn, params=params)
    finally:
        conn.close()


def scalar(sql, params=(), default=0.0):
    """Run a SELECT that returns a single value (e.g. SUM/COUNT)."""
    conn = get_conn()
    try:
        row = conn.execute(sql, params).fetchone()
        if row is None or row[0] is None:
            return default
        return row[0]
    finally:
        conn.close()


def init_db():
    """Create every table if it does not exist, then migrate any legacy CSV data."""
    conn = get_conn()
    try:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS expenses (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                date        TEXT    NOT NULL,
                category    TEXT    NOT NULL,
                amount      REAL    NOT NULL CHECK (amount > 0),
                description TEXT    NOT NULL
            );

            CREATE TABLE IF NOT EXISTS emergency_fund (
                id     INTEGER PRIMARY KEY AUTOINCREMENT,
                date   TEXT    NOT NULL,
                type   TEXT    NOT NULL CHECK (type IN ('Deposit', 'Withdrawal')),
                amount REAL    NOT NULL CHECK (amount > 0),
                note   TEXT
            );

            CREATE TABLE IF NOT EXISTS savings_goals (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                name          TEXT    NOT NULL,
                target_amount REAL    NOT NULL CHECK (target_amount > 0),
                target_date   TEXT    NOT NULL,
                created_on    TEXT    NOT NULL
            );

            CREATE TABLE IF NOT EXISTS savings_history (
                id      INTEGER PRIMARY KEY AUTOINCREMENT,
                goal_id INTEGER NOT NULL,
                date    TEXT    NOT NULL,
                amount  REAL    NOT NULL CHECK (amount > 0),
                note    TEXT,
                FOREIGN KEY (goal_id) REFERENCES savings_goals(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS investments (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                name          TEXT    NOT NULL,
                type          TEXT    NOT NULL,
                amount        REAL    NOT NULL CHECK (amount > 0),
                current_value REAL    NOT NULL CHECK (current_value >= 0),
                date          TEXT    NOT NULL,
                notes         TEXT
            );

            CREATE TABLE IF NOT EXISTS investment_goals (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                name          TEXT    NOT NULL,
                target_amount REAL    NOT NULL CHECK (target_amount > 0),
                target_date   TEXT    NOT NULL,
                created_on    TEXT    NOT NULL
            );
            """
        )
        conn.commit()
    finally:
        conn.close()

    migrate_legacy_csv()


def migrate_legacy_csv():
    """One-time import of the old expenses.csv into the relational database."""
    if not os.path.exists(LEGACY_CSV):
        return
    if scalar("SELECT COUNT(*) FROM expenses", default=0) > 0:
        return  # Database already has data; do not import twice.
    try:
        old = pd.read_csv(LEGACY_CSV)
    except Exception:
        return
    required = {"Date", "Category", "Amount", "Description"}
    if not required.issubset(set(old.columns)):
        return
    for _, row in old.iterrows():
        try:
            amount = float(row["Amount"])
        except (TypeError, ValueError):
            continue
        if amount <= 0:
            continue
        execute(
            "INSERT INTO expenses (date, category, amount, description) VALUES (?, ?, ?, ?)",
            (str(row["Date"]), str(row["Category"]), amount, str(row["Description"])),
        )


init_db()

# =============================================================================
# 3. CATEGORY MASTER LIST
# Every label used in train_data below MUST appear here, otherwise a correct
# prediction would be discarded by the membership check and silently lost.
# =============================================================================
CATEGORIES = [
    "Food",
    "Rent",
    "Travel",
    "Utilities",
    "Medical",
    "Entertainment",
    "Shopping",
    "Personal Care",
    "Education",
    "Investments",
    "Miscellaneous",
    "Other",
]

# =============================================================================
# 4. TRAINING CORPUS FOR THE NLP CLASSIFIER
# =============================================================================
train_data = [
    # ---------------- Food & Dining ----------------
    ("Swiggy biryani lunch dinner food restaurant pizza burger grocery milk vegetables snacks fruit pasta cafe", "Food"),
    ("Zomato order food dining cafe meals breakfast", "Food"),
    ("Supermarket grocery vegetables fruits milk bread eggs provisions store", "Food"),
    ("Starbucks coffee tea pastry bakery cafe snacks", "Food"),
    ("McDonalds KFC Dominoes burger pizza fast food meal", "Food"),
    ("Pub bar drinks alcohol beer wine cocktails dinner", "Food"),
    ("Blinkit Zepto Instamart BigBasket quick delivery grocery milk bread veggies", "Food"),
    ("Dunzo grocery pickup delivery fruits snacks items", "Food"),
    ("D-Mart Reliance Smart Hypermarket grocery household items vegetables", "Food"),
    ("Local kirana store cash grocery items rice dal oil sugar", "Food"),
    ("Dominos Pizza Hut Burger King meal lunch dinner", "Food"),
    ("Subway wrap salad sandwich healthy meal", "Food"),
    ("Haldiram Bikanervala sweets snacks namkeen chaat", "Food"),
    ("Canteen mess lunch thali dinner hostel food", "Food"),
    ("Ice cream naturals baskin robbins dessert", "Food"),
    ("Costa Coffee Blue Tokai cappuccino latte espresso cafe", "Food"),
    ("Barbeque Nation buffet dinner family lunch celebration restaurant", "Food"),
    ("Local bakery cake birthday pastry bread toast", "Food"),
    ("Juice center coconut water sugarcane juice fresh fruit shake", "Food"),
    ("Roadside tea stall chai samosa evening snacks", "Food"),
    ("Tiffin service monthly meal box lunch dinner delivery", "Food"),

    # ---------------- Travel & Commute ----------------
    ("Uber Ola auto metro travel cab train ticket flight petrol diesel bus commute", "Travel"),
    ("Shell HP Indian Oil fuel petrol diesel gas station refuel", "Travel"),
    ("IRCTC train ticket booking railway reservation", "Travel"),
    ("IndiGo Air India flight ticket airline baggage", "Travel"),
    ("Metro card recharge token smart card transit local train", "Travel"),
    ("Parking toll plaza fastag highway fee cab rental", "Travel"),
    ("Rapido bike taxi autorickshaw ride booking", "Travel"),
    ("InDrive cab ride local transport auto", "Travel"),
    ("RedBus bus ticket travel Volvo sleeper booking", "Travel"),
    ("Airport cab pickup drop outstation trip", "Travel"),
    ("Local train pass suburban railway ticket", "Travel"),
    ("Fastag recharge highway toll tax payment", "Travel"),
    ("Puncture repair car wash petrol pump air filling parking", "Travel"),
    ("Bike servicing car maintenance spare parts mechanic garage", "Travel"),
    ("Bike rental scooty hire vacation Goa trip fuel", "Travel"),
    ("Hotel stay resort booking accommodation room service trip", "Travel"),
    ("Airbnb homestay vacation rental trip booking", "Travel"),
    ("Toll booth cash payment highway express bridge fee", "Travel"),
    ("Flight seat selection baggage upgrade airline ticket fee", "Travel"),

    # ---------------- Utilities & Bills ----------------
    ("Electricity bill Wi-Fi recharge water utility internet broadband mobile recharge", "Utilities"),
    ("Airtel Jio Vi Vodafone postpaid prepaid fiber broadband recharge", "Utilities"),
    ("Power grid electricity board bill payment BESCOM TNEB", "Utilities"),
    ("Gas cylinder booking piped gas HP Gas Indane IOCL", "Utilities"),
    ("Municipal water bill tax maintenance utility", "Utilities"),
    ("Postpaid mobile bill Jio Airtel Vodafone Idea bill payment", "Utilities"),
    ("Broadband fiber connection bill ACT Hathway JioFiber Airtel Xstream", "Utilities"),
    ("Piped natural gas PNG bill Mahanagar Gas IGL Vadodara Gas", "Utilities"),
    ("Municipal corporation property tax water tax garbage collection", "Utilities"),
    ("DTH recharge Tata Play Airtel Digital TV Dish TV", "Utilities"),
    ("Google One iCloud Google Drive storage subscription monthly", "Utilities"),
    ("ChatGPT Plus OpenAI API subscription software tool billing", "Utilities"),
    ("GitHub Copilot Figma Notion software subscription productivity", "Utilities"),
    ("Domain name renewal GoDaddy Namecheap web hosting server bill", "Utilities"),
    ("LPG gas cylinder booking refill agency home delivery", "Utilities"),

    # ---------------- Medical & Healthcare ----------------
    ("Medicine doctor hospital checkup pharmacy medical lab test", "Medical"),
    ("Apollo Pharmacy Medplus 1mg healthcare medicines pills tablets", "Medical"),
    ("Dentist dental cavity root canal eye checkup clinic consultation surgeon", "Medical"),
    ("Blood test MRI X-ray ultrasound diagnostic laboratory scan report", "Medical"),
    ("Netmeds PharmEasy online medicine delivery order prescription", "Medical"),
    ("Pathology blood test home collection Thyrocare Lal PathLabs", "Medical"),
    ("Physiotherapy physiotherapist physio session clinic rehabilitation therapy", "Medical"),
    ("Hospital emergency room admission doctor consultation surgery operation bill", "Medical"),
    ("Optical store eyeglasses spectacles frames contact lenses prescription", "Medical"),
    ("Therapy session mental health counsellor psychologist psychiatrist consultation", "Medical"),
    ("Ayurvedic doctor clinic herbal medicine treatment", "Medical"),
    ("Homeopathy medicine clinic doctor prescription checkup", "Medical"),
    ("First aid kit bandage antiseptic ointment medical tape pharmacy", "Medical"),
    ("Vaccination injection booster dose immunisation shot", "Medical"),
    ("Health checkup package full body screening preventive diagnostics", "Medical"),

    # ---------------- Entertainment & Leisure ----------------
    ("Movie ticket Netflix Spotify gaming concert steam playstation party", "Entertainment"),
    ("Amazon Prime Disney Hotstar subscription streaming movie", "Entertainment"),
    ("BookMyShow cinema theater concert event pass tickets", "Entertainment"),
    ("PlayStation Steam Epic Games Xbox video game purchase", "Entertainment"),
    ("Bowling arcade theme park waterpark museum entry ticket", "Entertainment"),
    ("SonyLIV Zee5 Apple TV YouTube Premium subscription streaming", "Entertainment"),
    ("PVR Inox cinema ticket movie popcorn IMAX", "Entertainment"),
    ("Night club entry cover charge DJ party", "Entertainment"),
    ("Board game cafe escape room bowling alley arcade tokens", "Entertainment"),
    ("Standup comedy show concert festival pass live event", "Entertainment"),
    ("Kindle book ebook Audible audiobook fiction novel read", "Entertainment"),
    ("Museum art gallery exhibition heritage site entry ticket", "Entertainment"),
    ("Theme park roller coaster water park fast track pass", "Entertainment"),
    ("Twitch streamer donation Discord Nitro Patreon subscription", "Entertainment"),
    ("Puzzle hobby kit art supplies painting canvas", "Entertainment"),

    # ---------------- Rent & Housing ----------------
    ("House apartment room rent maintenance society charge lease", "Rent"),
    ("PG accommodation hostel monthly rent security deposit", "Rent"),
    ("Society maintenance charges flat apartment dues", "Rent"),
    ("Brokerage fee house shifting packers movers apartment shifting", "Rent"),
    ("Society maintenance monthly dues water charges lift repair", "Rent"),
    ("Electricity sub-meter bill payment to landlord roommate", "Rent"),
    ("Society generator diesel backup maintenance society bill", "Rent"),
    ("Society parking sticker gate pass annual fee", "Rent"),

    # ---------------- Shopping & Lifestyle ----------------
    ("Amazon Flipkart Myntra Ajio shopping clothes shoes electronics fashion", "Shopping"),
    ("Zara H&M Levis apparel clothing jeans shirt dress", "Shopping"),
    ("Decathlon sports shoes gym gear equipment fitness", "Shopping"),
    ("Mall shopping retail store electronics gadgets mobile laptop", "Shopping"),
    ("Nykaa makeup lipstick cosmetics online order", "Shopping"),
    ("Meesho online shopping budget clothes home decor", "Shopping"),
    ("Lifestyle Max Westside retail clothing store shopping", "Shopping"),
    ("Nike Puma Adidas sneakers sports shoes apparel", "Shopping"),
    ("Croma Reliance Digital electronics laptop phone charger gadget", "Shopping"),
    ("IKEA home decor furniture mattress bedsheet pillow", "Shopping"),
    ("Apple store iPhone iPad Macbook accessories charger cable", "Shopping"),
    ("Samsung Xiaomi OnePlus mobile phone repair screen guard", "Shopping"),
    ("Thrift store vintage clothing second-hand market shopping", "Shopping"),
    ("Jewellery gold silver ring necklace gift wedding purchase", "Shopping"),
    ("Watch smart watch strap Fossil Fastrack Titan shopping", "Shopping"),
    ("Hardware store screws hammer drill tools home repair DIY", "Shopping"),

    # ---------------- Personal Care & Grooming ----------------
    ("Gym membership fitness center yoga crossfit workout", "Personal Care"),
    ("Salon haircut spa facial grooming parlour massage", "Personal Care"),
    ("Skincare shampoo cosmetics makeup perfume trimmer", "Personal Care"),
    ("Lakme Salon Naturals Jawed Habib haircut beard trimming", "Personal Care"),
    ("Gold Gym Cult fit fitness workout pass membership", "Personal Care"),
    ("Spa massage therapy relaxation wellness center", "Personal Care"),
    ("Shampoo soap toothpaste deodorant grooming kit", "Personal Care"),
    ("Protein powder creatine whey supplement nutrition gym", "Personal Care"),
    ("Laundry dry cleaning iron clothes dhobi service", "Personal Care"),
    ("Tailor blouse stitching pant alteration custom clothing", "Personal Care"),
    ("Shoe repair cobbler heel fixing leather polish", "Personal Care"),

    # ---------------- Education & Learning ----------------
    ("Udemy Coursera college school tuition fee books course university", "Education"),
    ("Exam fee tuition classes coaching stationery notebook pen pencil", "Education"),
    ("LinkedIn Learning masterclass certificate online course", "Education"),
    ("School fees semester fee admission college installment", "Education"),
    ("Stationery notebook pen pencil eraser calculator printer ink books", "Education"),
    ("Library membership competitive exam book study material", "Education"),

    # ---------------- Investments & Finance ----------------
    ("Mutual fund SIP stock market investment Zerodha Groww shares crypto", "Investments"),
    ("Fixed deposit recurring deposit PPF NPS retirement savings", "Investments"),
    ("Coin Kuvera systematic investment plan portfolio", "Investments"),
    ("Stock market shares equity intraday trading brokerage fee", "Investments"),
    ("Credit card bill payment EMI loan debit bank charges", "Investments"),
    ("Term insurance health insurance premium LIC policy payment", "Investments"),
    ("Personal loan EMI credit card foreclosure interest", "Investments"),
    ("Home loan installment bank transfer mortgage principal", "Investments"),
    ("Car insurance vehicle comprehensive policy renewal payment", "Investments"),
    ("Demat account annual maintenance charge AMC brokerage fee", "Investments"),
    ("Foreign exchange forex card currency conversion markup fee", "Investments"),

    # ---------------- Miscellaneous & Transfers ----------------
    ("ATM withdrawal cash transfer friend UPI peer lending gift", "Miscellaneous"),
    ("Donation charity temple church NGO welfare contribution", "Miscellaneous"),
    ("Gift birthday party present wedding shagun envelope", "Miscellaneous"),
    ("Pet food Pedigree Whiskas cat litter dog treats pet shop", "Miscellaneous"),
    ("Vet clinic vaccination dog grooming pet spa pet checkup", "Miscellaneous"),
    ("Office farewell contribution team party pool collection gift", "Miscellaneous"),
    ("Tip delivery boy restaurant waiter valet parking cash tip", "Miscellaneous"),
    ("Lost wallet replacement card fee government ID reissue", "Miscellaneous"),
]

# Safety net: a label that is not in CATEGORIES can never be selected in the UI.
_unknown = {label for _, label in train_data} - set(CATEGORIES)
if _unknown:
    st.error(f"Training labels missing from CATEGORIES: {sorted(_unknown)}")


@st.cache_resource
def build_classifier(data):
    """Fit the TF-IDF + Naive Bayes pipeline once and cache it across reruns."""
    X, y = zip(*data)
    pipeline = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, lowercase=True),
        MultinomialNB(alpha=0.1),
    )
    pipeline.fit(X, y)
    return pipeline


ai_classifier = build_classifier(tuple(train_data))


def predict_category(text):
    """Return the ML-predicted category for a free-text description."""
    if not text.strip():
        return "Other"
    try:
        label = ai_classifier.predict([text])[0]
        return label if label in CATEGORIES else "Other"
    except Exception:
        return "Other"


# =============================================================================
# 5. SHARED HELPERS
# =============================================================================
def total_expenses():
    return float(scalar("SELECT SUM(amount) FROM expenses"))


def emergency_balance():
    deposits = float(scalar("SELECT SUM(amount) FROM emergency_fund WHERE type = 'Deposit'"))
    withdrawals = float(scalar("SELECT SUM(amount) FROM emergency_fund WHERE type = 'Withdrawal'"))
    return deposits - withdrawals


def total_saved_in_goals():
    return float(scalar("SELECT SUM(amount) FROM savings_history"))


def investment_totals():
    invested = float(scalar("SELECT SUM(amount) FROM investments"))
    current = float(scalar("SELECT SUM(current_value) FROM investments"))
    return invested, current, current - invested


def days_left(target_date_str):
    """Whole days between today and an ISO date string (negative if overdue)."""
    try:
        target = datetime.strptime(target_date_str, "%Y-%m-%d").date()
    except ValueError:
        return None
    return (target - date.today()).days


def safe_progress(value, total):
    """Clamp a ratio into the 0.0-1.0 range that st.progress requires."""
    if total <= 0:
        return 0.0
    return max(0.0, min(1.0, value / total))


def show_pie(values, labels, colors=None, title=None):
    """Draw a pie chart, skipping gracefully when there is nothing to plot."""
    clean = [(v, l) for v, l in zip(values, labels) if v and v > 0]
    if not clean:
        st.info("Not enough data yet to draw this chart.")
        return
    vals = [c[0] for c in clean]
    labs = [c[1] for c in clean]
    used_colors = None
    if colors:
        mapping = dict(zip(labels, colors))
        used_colors = [mapping[l] for l in labs]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie(vals, labels=labs, autopct="%1.1f%%", startangle=90, colors=used_colors)
    ax.axis("equal")
    if title:
        ax.set_title(title)
    st.pyplot(fig)
    plt.close(fig)


# =============================================================================
# 6. NAVIGATION
# =============================================================================
st.title("💰 Mocker — AI-Infused Expense & Financial Manager")

st.sidebar.header("Navigation")
menu = st.sidebar.radio(
    "Select Option",
    [
        "Dashboard",
        "Add Expense",
        "View & Analyze",
        "Money Management",
        "Investment Manager",
        "AI Financial News",
        "Settings",
    ],
)
st.sidebar.caption("Storage: SQLite (mocker.db)")
st.sidebar.caption("AI: TF-IDF + Multinomial Naive Bayes")

# =============================================================================
# MODULE 1 - DASHBOARD
# =============================================================================
if menu == "Dashboard":
    st.subheader("Financial Overview")

    spent = total_expenses()
    ef_balance = emergency_balance()
    invested, current_value, returns = investment_totals()
    saved = total_saved_in_goals()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Expenditure", money(spent))
    c2.metric("Emergency Fund", money(ef_balance))
    c3.metric("Saved Towards Goals", money(saved))
    c4.metric("Portfolio Value", money(current_value), delta=money(returns))

    st.divider()
    st.subheader("⏳ Countdown to Your Financial Goals")

    s_goals = query_df("SELECT * FROM savings_goals ORDER BY target_date ASC")
    i_goals = query_df("SELECT * FROM investment_goals ORDER BY target_date ASC")

    if s_goals.empty and i_goals.empty:
        st.info(
            "No goals set yet. Create a Savings Goal under **Money Management** or an "
            "Investment Goal under **Investment Manager** to start the countdown."
        )
    else:
        for _, goal in s_goals.iterrows():
            saved_amt = float(
                scalar("SELECT SUM(amount) FROM savings_history WHERE goal_id = ?", (int(goal["id"]),))
            )
            left = days_left(goal["target_date"])
            col_a, col_b = st.columns([3, 1])
            with col_a:
                st.markdown(f"**🏦 {goal['name']}** &nbsp;·&nbsp; Savings Goal")
                st.progress(
                    safe_progress(saved_amt, float(goal["target_amount"])),
                    text=f"{money(saved_amt)} of {money(float(goal['target_amount']))}",
                )
            with col_b:
                if left is None:
                    st.metric("Days Left", "—")
                elif left > 0:
                    st.metric("Days Left", f"{left}")
                elif left == 0:
                    st.metric("Days Left", "Today")
                else:
                    st.metric("Days Left", f"{abs(left)} overdue")

        for _, goal in i_goals.iterrows():
            invested_total, _, _ = investment_totals()
            left = days_left(goal["target_date"])
            col_a, col_b = st.columns([3, 1])
            with col_a:
                st.markdown(f"**📈 {goal['name']}** &nbsp;·&nbsp; Investment Goal")
                st.progress(
                    safe_progress(invested_total, float(goal["target_amount"])),
                    text=f"{money(invested_total)} of {money(float(goal['target_amount']))}",
                )
            with col_b:
                if left is None:
                    st.metric("Days Left", "—")
                elif left > 0:
                    st.metric("Days Left", f"{left}")
                elif left == 0:
                    st.metric("Days Left", "Today")
                else:
                    st.metric("Days Left", f"{abs(left)} overdue")

    st.divider()
    st.subheader("Recent Expenses")
    recent = query_df("SELECT date, category, amount, description FROM expenses ORDER BY id DESC LIMIT 8")
    if recent.empty:
        st.info("No expenses logged yet. Head to **Add Expense** to record your first one.")
    else:
        recent.columns = ["Date", "Category", "Amount", "Description"]
        st.dataframe(recent, hide_index=True)

# =============================================================================
# MODULE 2 - ADD EXPENSE (live AI categorisation)
# =============================================================================
elif menu == "Add Expense":
    st.subheader("Log a New Expense with Live AI Categorization")

    exp_date = st.date_input("Date", value=date.today())
    description = st.text_input(
        "Description",
        placeholder="Type naturally, e.g. 'Physiotherapy session at the clinic'",
    )

    suggested_category = predict_category(description)

    if description.strip():
        st.markdown(f"**Category:** `{suggested_category}`")
    else:
        st.caption("Type a description and the model will pick a category for you.")

    default_idx = (
        CATEGORIES.index(suggested_category)
        if suggested_category in CATEGORIES
        else CATEGORIES.index("Other")
    )
    category = st.selectbox("Confirm or Override Category", CATEGORIES, index=default_idx)
    amount = st.number_input(f"Amount ({RUPEE})", min_value=1.0, value=100.0, format="%.2f")

    if st.button("Add Expense", type="primary"):
        if not description.strip():
            st.warning("Please enter a description so the model can categorise it.")
        else:
            execute(
                "INSERT INTO expenses (date, category, amount, description) VALUES (?, ?, ?, ?)",
                (str(exp_date), category, float(amount), description.strip()),
            )
            st.success(f"Expense of {money(amount)} saved under **{category}**.")

# =============================================================================
# MODULE 3 - VIEW & ANALYZE (CRUD + analytics)
# =============================================================================
elif menu == "View & Analyze":
    st.subheader("Expense History & Analytics")

    df = query_df("SELECT id, date, category, amount, description FROM expenses ORDER BY date DESC, id DESC")

    if df.empty:
        st.info("No expenses recorded yet. Switch to **Add Expense** from the sidebar to get started.")
    else:
        display = df.copy()
        display.columns = ["ID", "Date", "Category", "Amount", "Description"]

        all_cats = sorted(display["Category"].unique().tolist())
        chosen = st.multiselect("Filter by category", all_cats, default=all_cats)
        filtered = display[display["Category"].isin(chosen)] if chosen else display

        st.dataframe(filtered, hide_index=True)

        m1, m2, m3 = st.columns(3)
        m1.metric("Total Expenditure", money(filtered["Amount"].sum()))
        m2.metric("Number of Entries", f"{len(filtered)}")
        m3.metric(
            "Average per Entry",
            money(filtered["Amount"].mean()) if len(filtered) else money(0),
        )

        st.divider()
        st.subheader("Spending by Category")

        if filtered.empty:
            st.info("No rows match the current filter.")
        else:
            totals = filtered.groupby("Category")["Amount"].sum().sort_values(ascending=False)

            chart_col, pie_col = st.columns(2)
            with chart_col:
                fig, ax = plt.subplots(figsize=(6, 4))
                totals.plot(kind="bar", ax=ax, color="skyblue")
                ax.set_ylabel("Amount (Rs)")
                ax.set_xlabel("Category")
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            with pie_col:
                show_pie(totals.tolist(), totals.index.tolist(), title="Share of Total Spending")

        st.divider()
        st.subheader("Manage Entries")

        id_options = display["ID"].tolist()
        row_id = st.selectbox(
            "Select an entry",
            id_options,
            format_func=lambda i: (
                f"#{i} · {display.loc[display['ID'] == i, 'Date'].values[0]} · "
                f"{display.loc[display['ID'] == i, 'Category'].values[0]} · "
                f"{RUPEE}{display.loc[display['ID'] == i, 'Amount'].values[0]:,.2f}"
            ),
        )

        row = df[df["id"] == row_id].iloc[0]

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Delete this entry**")
            st.caption(f"{row['description']} — {money(float(row['amount']))}")
            if st.button("Delete Selected Entry"):
                execute("DELETE FROM expenses WHERE id = ?", (int(row_id),))
                st.success(f"Entry #{row_id} deleted.")
                st.rerun()

        with col2:
            with st.form("edit_form"):
                st.markdown(f"**Editing entry #{row_id}**")

                current_cat = str(row["category"])
                cat_idx = (
                    CATEGORIES.index(current_cat)
                    if current_cat in CATEGORIES
                    else CATEGORIES.index("Other")
                )

                try:
                    current_date = datetime.strptime(str(row["date"]), "%Y-%m-%d").date()
                except ValueError:
                    current_date = date.today()

                new_date = st.date_input("Date", value=current_date)
                new_cat = st.selectbox("Category", CATEGORIES, index=cat_idx)
                new_amt = st.number_input(
                    f"Amount ({RUPEE})", value=float(row["amount"]), min_value=1.0, format="%.2f"
                )
                new_desc = st.text_input("Description", value=str(row["description"]))

                if st.form_submit_button("Update Entry"):
                    execute(
                        "UPDATE expenses SET date = ?, category = ?, amount = ?, description = ? WHERE id = ?",
                        (str(new_date), new_cat, float(new_amt), new_desc.strip(), int(row_id)),
                    )
                    st.success("Entry updated successfully.")
                    st.rerun()

# =============================================================================
# MODULE 4 - MONEY MANAGEMENT (Emergency Fund + Savings Goals)
# =============================================================================
elif menu == "Money Management":
    st.subheader("Money Management")

    tab_ef, tab_goals = st.tabs(["🚨 Emergency Fund", "🎯 Savings Goals"])

    # ---------------------------------------------------------------- Emergency
    with tab_ef:
        balance = emergency_balance()
        deposits = float(scalar("SELECT SUM(amount) FROM emergency_fund WHERE type = 'Deposit'"))
        withdrawals = float(scalar("SELECT SUM(amount) FROM emergency_fund WHERE type = 'Withdrawal'"))

        e1, e2, e3 = st.columns(3)
        e1.metric("Current Balance", money(balance))
        e2.metric("Total Deposited", money(deposits))
        e3.metric("Total Withdrawn", money(withdrawals))

        st.markdown("#### Record a Transaction")
        with st.form("ef_form"):
            f1, f2, f3 = st.columns(3)
            with f1:
                ef_type = st.radio("Type", ["Deposit", "Withdrawal"], horizontal=True)
            with f2:
                ef_amount = st.number_input(
                    f"Amount ({RUPEE})", min_value=1.0, value=1000.0, format="%.2f", key="ef_amt"
                )
            with f3:
                ef_date = st.date_input("Date", value=date.today(), key="ef_date")
            ef_note = st.text_input("Note (optional)", placeholder="e.g. Monthly transfer from salary")

            if st.form_submit_button("Save Transaction", type="primary"):
                if ef_type == "Withdrawal" and ef_amount > balance:
                    st.error(
                        f"You cannot withdraw {money(ef_amount)} — the fund only holds {money(balance)}."
                    )
                else:
                    execute(
                        "INSERT INTO emergency_fund (date, type, amount, note) VALUES (?, ?, ?, ?)",
                        (str(ef_date), ef_type, float(ef_amount), ef_note.strip()),
                    )
                    st.success(f"{ef_type} of {money(ef_amount)} recorded.")
                    st.rerun()

        st.markdown("#### Transaction History")
        ef_df = query_df("SELECT id, date, type, amount, note FROM emergency_fund ORDER BY date ASC, id ASC")

        if ef_df.empty:
            st.info("No emergency fund activity yet. Record your first deposit above.")
        else:
            history = ef_df.copy()
            signed = history.apply(
                lambda r: r["amount"] if r["type"] == "Deposit" else -r["amount"], axis=1
            )
            history["Running Balance"] = signed.cumsum()
            history.columns = ["ID", "Date", "Type", "Amount", "Note", "Running Balance"]

            st.dataframe(
                history.sort_values("ID", ascending=False),
                hide_index=True,
            )

            fig, ax = plt.subplots(figsize=(7, 3.5))
            ax.plot(range(len(history)), history["Running Balance"], marker="o", color="seagreen")
            ax.set_xticks(range(len(history)))
            ax.set_xticklabels(history["Date"], rotation=45, ha="right")
            ax.set_ylabel("Balance (Rs)")
            ax.set_title("Emergency Fund Growth")
            ax.grid(alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            del_id = st.selectbox("Delete a transaction", ef_df["id"].tolist(), key="ef_del")
            if st.button("Delete Transaction", key="ef_del_btn"):
                execute("DELETE FROM emergency_fund WHERE id = ?", (int(del_id),))
                st.success("Transaction deleted.")
                st.rerun()

    # -------------------------------------------------------------- Savings Goals
    with tab_goals:
        st.markdown("#### Create a Savings Goal")
        with st.form("goal_form"):
            g1, g2, g3 = st.columns(3)
            with g1:
                g_name = st.text_input("Goal Name", placeholder="e.g. New Laptop")
            with g2:
                g_target = st.number_input(
                    f"Target Amount ({RUPEE})", min_value=1.0, value=5000.0, format="%.2f"
                )
            with g3:
                g_date = st.date_input("Target Date", value=date.today())

            if st.form_submit_button("Create Goal", type="primary"):
                if not g_name.strip():
                    st.warning("Please give the goal a name.")
                else:
                    execute(
                        "INSERT INTO savings_goals (name, target_amount, target_date, created_on) "
                        "VALUES (?, ?, ?, ?)",
                        (g_name.strip(), float(g_target), str(g_date), str(date.today())),
                    )
                    st.success(f"Goal '{g_name.strip()}' created.")
                    st.rerun()

        goals = query_df("SELECT * FROM savings_goals ORDER BY target_date ASC")

        if goals.empty:
            st.info("No savings goals yet. Create one above to unlock the progress chart and history.")
        else:
            st.divider()
            goal_id = st.selectbox(
                "Select a goal",
                goals["id"].tolist(),
                format_func=lambda i: goals.loc[goals["id"] == i, "name"].values[0],
            )
            goal = goals[goals["id"] == goal_id].iloc[0]

            target_amount = float(goal["target_amount"])
            saved_amt = float(
                scalar("SELECT SUM(amount) FROM savings_history WHERE goal_id = ?", (int(goal_id),))
            )
            remaining = max(target_amount - saved_amt, 0.0)
            left = days_left(goal["target_date"])

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Target", money(target_amount))
            k2.metric("Saved So Far", money(saved_amt))
            k3.metric("Remaining", money(remaining))
            if left is None:
                k4.metric("Days Left", "—")
            elif left > 0:
                k4.metric("Days Left", f"{left}")
            elif left == 0:
                k4.metric("Days Left", "Today")
            else:
                k4.metric("Days Left", f"{abs(left)} overdue")

            st.progress(
                safe_progress(saved_amt, target_amount),
                text=f"{safe_progress(saved_amt, target_amount) * 100:.1f}% complete",
            )

            if remaining > 0 and left is not None and left > 0:
                st.caption(
                    f"To hit this goal on time you need about {money(remaining / left)} per day "
                    f"for the next {left} days."
                )
            elif remaining <= 0:
                st.success("🎉 Goal achieved! You have reached the target amount.")

            chart_col, form_col = st.columns(2)

            with chart_col:
                show_pie(
                    [saved_amt, remaining],
                    ["Saved", "Remaining"],
                    colors=["#2e8b57", "#d3d3d3"],
                    title=str(goal["name"]),
                )

            with form_col:
                st.markdown("#### Add a Contribution")
                with st.form("contrib_form"):
                    c_amount = st.number_input(
                        f"Amount ({RUPEE})", min_value=1.0, value=500.0, format="%.2f", key="c_amt"
                    )
                    c_date = st.date_input("Date", value=date.today(), key="c_date")
                    c_note = st.text_input("Note (optional)", key="c_note")
                    if st.form_submit_button("Add to Savings", type="primary"):
                        execute(
                            "INSERT INTO savings_history (goal_id, date, amount, note) VALUES (?, ?, ?, ?)",
                            (int(goal_id), str(c_date), float(c_amount), c_note.strip()),
                        )
                        st.success(f"{money(c_amount)} added to '{goal['name']}'.")
                        st.rerun()

            st.markdown("#### Savings History")
            hist = query_df(
                "SELECT id, date, amount, note FROM savings_history WHERE goal_id = ? ORDER BY date DESC, id DESC",
                (int(goal_id),),
            )
            if hist.empty:
                st.info("No contributions recorded for this goal yet.")
            else:
                hist.columns = ["ID", "Date", "Amount", "Note"]
                st.dataframe(hist, hide_index=True)

                del_contrib = st.selectbox(
                    "Delete a contribution", hist["ID"].tolist(), key="contrib_del"
                )
                if st.button("Delete Contribution", key="contrib_del_btn"):
                    execute("DELETE FROM savings_history WHERE id = ?", (int(del_contrib),))
                    st.success("Contribution deleted.")
                    st.rerun()

            st.divider()
            if st.button("🗑️ Delete This Entire Goal", key="goal_del_btn"):
                execute("DELETE FROM savings_history WHERE goal_id = ?", (int(goal_id),))
                execute("DELETE FROM savings_goals WHERE id = ?", (int(goal_id),))
                st.success("Goal and its history deleted.")
                st.rerun()

# =============================================================================
# MODULE 5 - INVESTMENT MANAGER
# =============================================================================
elif menu == "Investment Manager":
    st.subheader("Investment Manager")

    tab_port, tab_analysis, tab_goals = st.tabs(
        ["💼 My Portfolio", "📊 Invested vs Earned vs Spent", "🎯 Investment Goals"]
    )

    INVESTMENT_TYPES = [
        "Mutual Fund / SIP",
        "Stocks / Equity",
        "Fixed Deposit",
        "Recurring Deposit",
        "PPF / NPS",
        "Gold",
        "Crypto",
        "Real Estate",
        "Other",
    ]

    # ------------------------------------------------------------------ Portfolio
    with tab_port:
        st.markdown("#### Add an Investment")
        with st.form("inv_form"):
            i1, i2 = st.columns(2)
            with i1:
                inv_name = st.text_input("Investment Name", placeholder="e.g. Nifty 50 Index Fund")
                inv_type = st.selectbox("Type", INVESTMENT_TYPES)
                inv_date = st.date_input("Date Invested", value=date.today(), key="inv_date")
            with i2:
                inv_amount = st.number_input(
                    f"Amount Invested ({RUPEE})", min_value=1.0, value=5000.0, format="%.2f"
                )
                inv_current = st.number_input(
                    f"Current Value ({RUPEE})", min_value=0.0, value=5000.0, format="%.2f"
                )
                inv_notes = st.text_input("Notes (optional)", placeholder="e.g. Monthly SIP via Groww")

            if st.form_submit_button("Add Investment", type="primary"):
                if not inv_name.strip():
                    st.warning("Please enter the investment name.")
                else:
                    execute(
                        "INSERT INTO investments (name, type, amount, current_value, date, notes) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        (
                            inv_name.strip(),
                            inv_type,
                            float(inv_amount),
                            float(inv_current),
                            str(inv_date),
                            inv_notes.strip(),
                        ),
                    )
                    st.success(f"'{inv_name.strip()}' added to your portfolio.")
                    st.rerun()

        st.markdown("#### Where Your Money Is Invested")
        inv_df = query_df(
            "SELECT id, name, type, amount, current_value, date, notes FROM investments ORDER BY date DESC, id DESC"
        )

        if inv_df.empty:
            st.info("No investments recorded yet. Add your first holding above.")
        else:
            table = inv_df.copy()
            table["Returns"] = table["current_value"] - table["amount"]
            table["Return %"] = (table["Returns"] / table["amount"] * 100).round(2)
            table = table[["id", "name", "type", "amount", "current_value", "Returns", "Return %", "date", "notes"]]
            table.columns = [
                "ID", "Name", "Type", "Invested", "Current Value", "Returns", "Return %", "Date", "Notes",
            ]
            st.dataframe(table, hide_index=True)

            invested, current_value, returns = investment_totals()
            p1, p2, p3 = st.columns(3)
            p1.metric("Total Invested", money(invested))
            p2.metric("Current Value", money(current_value))
            p3.metric("Net Returns", money(returns), delta=f"{(returns / invested * 100):.2f}%" if invested else None)

            st.markdown("#### Allocation by Type")
            by_type = inv_df.groupby("type")["amount"].sum()
            show_pie(by_type.tolist(), by_type.index.tolist(), title="Portfolio Allocation")

            st.divider()
            st.markdown("#### Manage Holdings")
            sel_id = st.selectbox(
                "Select a holding",
                inv_df["id"].tolist(),
                format_func=lambda i: f"#{i} · {inv_df.loc[inv_df['id'] == i, 'name'].values[0]}",
            )
            holding = inv_df[inv_df["id"] == sel_id].iloc[0]

            u1, u2 = st.columns(2)
            with u1:
                with st.form("inv_update_form"):
                    st.markdown(f"**Update '{holding['name']}'**")
                    upd_value = st.number_input(
                        f"Current Value ({RUPEE})",
                        min_value=0.0,
                        value=float(holding["current_value"]),
                        format="%.2f",
                    )
                    upd_notes = st.text_input("Notes", value=str(holding["notes"] or ""))
                    if st.form_submit_button("Update Holding"):
                        execute(
                            "UPDATE investments SET current_value = ?, notes = ? WHERE id = ?",
                            (float(upd_value), upd_notes.strip(), int(sel_id)),
                        )
                        st.success("Holding updated.")
                        st.rerun()
            with u2:
                st.markdown("**Remove a holding**")
                st.caption(f"{holding['name']} — invested {money(float(holding['amount']))}")
                if st.button("Delete Holding", key="inv_del_btn"):
                    execute("DELETE FROM investments WHERE id = ?", (int(sel_id),))
                    st.success("Holding deleted.")
                    st.rerun()

    # ------------------------------------------------------------------ Analysis
    with tab_analysis:
        invested, current_value, returns = investment_totals()
        spent = total_expenses()

        a1, a2, a3 = st.columns(3)
        a1.metric("Money Invested", money(invested))
        a2.metric("Money Earned (Returns)", money(returns))
        a3.metric("Money Spent (Expenses)", money(spent))

        st.markdown("#### Invested vs Earned vs Expenditure")

        if invested <= 0 and spent <= 0 and returns == 0:
            st.info("Add some expenses and investments first — the comparison chart needs data.")
        else:
            pie_col, bar_col = st.columns(2)

            with pie_col:
                if returns < 0:
                    st.caption(
                        "Returns are currently negative, so they are excluded from the pie "
                        "(a pie chart cannot display a negative slice). See the bar chart."
                    )
                show_pie(
                    [invested, max(returns, 0.0), spent],
                    ["Invested", "Earned", "Expenditure"],
                    colors=["#4c72b0", "#2e8b57", "#c44e52"],
                    title="Money Distribution",
                )

            with bar_col:
                labels = ["Invested", "Earned", "Expenditure"]
                values = [invested, returns, spent]
                colors = ["#4c72b0", "#2e8b57" if returns >= 0 else "#c44e52", "#c44e52"]

                fig, ax = plt.subplots(figsize=(5, 5))
                bars = ax.bar(labels, values, color=colors)
                ax.set_ylabel("Amount (Rs)")
                ax.axhline(0, color="black", linewidth=0.8)
                ax.set_title("Absolute Comparison")
                for bar, value in zip(bars, values):
                    ax.annotate(
                        f"{value:,.0f}",
                        xy=(bar.get_x() + bar.get_width() / 2, value),
                        xytext=(0, 4 if value >= 0 else -14),
                        textcoords="offset points",
                        ha="center",
                        fontsize=9,
                    )
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

            outflow = invested + spent
            if outflow > 0:
                st.caption(
                    f"Of every {RUPEE}100 leaving your account, "
                    f"{RUPEE}{invested / outflow * 100:.1f} goes into investments and "
                    f"{RUPEE}{spent / outflow * 100:.1f} into expenses."
                )

    # ------------------------------------------------------------------- Goals
    with tab_goals:
        st.markdown("#### Set an Investment Goal")
        with st.form("inv_goal_form"):
            v1, v2, v3 = st.columns(3)
            with v1:
                ig_name = st.text_input("Goal Name", placeholder="e.g. 1 Lakh Portfolio")
            with v2:
                ig_target = st.number_input(
                    f"Target Amount ({RUPEE})", min_value=1.0, value=100000.0, format="%.2f"
                )
            with v3:
                ig_date = st.date_input("Target Date", value=date.today(), key="ig_date")

            if st.form_submit_button("Create Investment Goal", type="primary"):
                if not ig_name.strip():
                    st.warning("Please give the goal a name.")
                else:
                    execute(
                        "INSERT INTO investment_goals (name, target_amount, target_date, created_on) "
                        "VALUES (?, ?, ?, ?)",
                        (ig_name.strip(), float(ig_target), str(ig_date), str(date.today())),
                    )
                    st.success(f"Investment goal '{ig_name.strip()}' created.")
                    st.rerun()

        ig_df = query_df("SELECT * FROM investment_goals ORDER BY target_date ASC")

        if ig_df.empty:
            st.info("No investment goals yet. Create one above and it will appear on the Dashboard countdown.")
        else:
            invested, _, _ = investment_totals()
            st.markdown("#### Your Investment Goals")

            for _, g in ig_df.iterrows():
                target = float(g["target_amount"])
                left = days_left(g["target_date"])
                remaining = max(target - invested, 0.0)

                with st.container(border=True):
                    gc1, gc2, gc3 = st.columns([2, 1, 1])
                    gc1.markdown(f"**{g['name']}**")
                    gc1.caption(f"Target date: {g['target_date']}")
                    gc2.metric("Target", money(target))
                    if left is None:
                        gc3.metric("Days Left", "—")
                    elif left > 0:
                        gc3.metric("Days Left", f"{left}")
                    elif left == 0:
                        gc3.metric("Days Left", "Today")
                    else:
                        gc3.metric("Days Left", f"{abs(left)} overdue")

                    st.progress(
                        safe_progress(invested, target),
                        text=f"{money(invested)} invested of {money(target)} · {money(remaining)} to go",
                    )

            del_goal = st.selectbox(
                "Delete an investment goal",
                ig_df["id"].tolist(),
                format_func=lambda i: ig_df.loc[ig_df["id"] == i, "name"].values[0],
                key="ig_del",
            )
            if st.button("Delete Investment Goal", key="ig_del_btn"):
                execute("DELETE FROM investment_goals WHERE id = ?", (int(del_goal),))
                st.success("Investment goal deleted.")
                st.rerun()

# =============================================================================
# MODULE 6 - AI FINANCIAL NEWS
# =============================================================================
elif menu == "AI Financial News":
    st.subheader("🤖 AI-Powered Investment & Financial News Feed")
    st.markdown("Synthesised market insights and portfolio strategies presented for your profile.")

    st.caption(
        "Note: this module serves a curated, offline insight set. Live market data would "
        "require a financial news API key, which is listed under future scope."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.info(
            "**Market Trend: Tech & AI Index**\n\nThe Indian tech sector shows bullish momentum "
            "heading into Q3. Analysts recommend systematic investment plans (SIPs) in diversified "
            "index funds rather than lump-sum trading."
        )
        st.success(
            "**Investment Tip**\n\nAllocate at least 20% of monthly savings into liquid debt funds "
            "to maintain a healthy emergency buffer while staying ahead of inflation."
        )
    with col2:
        st.warning(
            "**Macroeconomic Alert**\n\nInterest rates are steadying. Fixed deposit rates remain "
            "attractive for low-risk capital preservation, but equity mutual funds offer superior "
            "long-term compounding for students building a long horizon portfolio."
        )
        st.error(
            "**Risk Watch**\n\nHigh-volatility assets such as crypto should generally stay under "
            "5-10% of a beginner portfolio. Never invest your emergency fund in volatile instruments."
        )

    st.divider()
    st.markdown("#### Personalised Insight")

    invested, current_value, returns = investment_totals()
    ef_balance = emergency_balance()
    spent = total_expenses()

    if invested == 0 and spent == 0:
        st.info("Log a few expenses and investments and this section will analyse your own numbers.")
    else:
        monthly_cushion = ef_balance / spent if spent > 0 else 0
        if ef_balance <= 0:
            st.warning(
                "Your emergency fund is empty. Financial planners suggest building 3-6 months of "
                "expenses before increasing market exposure."
            )
        else:
            st.write(
                f"Your emergency fund of **{money(ef_balance)}** covers roughly "
                f"**{monthly_cushion:.2f}x** your total recorded expenditure of {money(spent)}."
            )
        if invested > 0:
            st.write(
                f"Your portfolio stands at **{money(current_value)}** against **{money(invested)}** "
                f"invested — a net return of **{money(returns)}**."
            )

    if st.button("Refresh AI Market Insights"):
        st.toast("Latest market insights refreshed.")

# =============================================================================
# MODULE 7 - SETTINGS
# =============================================================================
elif menu == "Settings":
    st.subheader("Settings & Data Management")

    st.markdown("#### Demo Data")
    st.caption(
        "Loads a realistic sample dataset so every chart and countdown is populated "
        "during your presentation. Only works when the database is empty."
    )

    if st.button("Load Demo Data"):
        if scalar("SELECT COUNT(*) FROM expenses", default=0) > 0:
            st.warning("The database already contains data. Reset it first if you want a clean demo.")
        else:
            demo_expenses = [
                ("2026-09-01", "Rent", 8500.0, "Monthly PG rent payment"),
                ("2026-09-02", "Food", 340.0, "Swiggy biryani order"),
                ("2026-09-03", "Travel", 120.0, "Uber to college"),
                ("2026-09-04", "Education", 450.0, "Notebook pen and stationery"),
                ("2026-09-05", "Medical", 900.0, "Physiotherapy session at clinic"),
                ("2026-09-06", "Utilities", 799.0, "Jio broadband monthly bill"),
                ("2026-09-08", "Entertainment", 500.0, "PVR movie ticket with friends"),
                ("2026-09-10", "Shopping", 2200.0, "Myntra shoes order"),
                ("2026-09-11", "Personal Care", 300.0, "Salon haircut"),
                ("2026-09-12", "Food", 1150.0, "BigBasket grocery for the month"),
                ("2026-09-13", "Travel", 650.0, "IRCTC train ticket home"),
                ("2026-09-14", "Miscellaneous", 200.0, "Birthday gift for friend"),
            ]
            for row in demo_expenses:
                execute(
                    "INSERT INTO expenses (date, category, amount, description) VALUES (?, ?, ?, ?)",
                    row,
                )

            demo_ef = [
                ("2026-07-05", "Deposit", 5000.0, "Initial emergency corpus"),
                ("2026-08-05", "Deposit", 3000.0, "Monthly transfer"),
                ("2026-08-20", "Withdrawal", 1500.0, "Urgent laptop repair"),
                ("2026-09-05", "Deposit", 3000.0, "Monthly transfer"),
            ]
            for row in demo_ef:
                execute(
                    "INSERT INTO emergency_fund (date, type, amount, note) VALUES (?, ?, ?, ?)", row
                )

            goal_id = execute(
                "INSERT INTO savings_goals (name, target_amount, target_date, created_on) VALUES (?, ?, ?, ?)",
                ("New Laptop", 55000.0, "2026-12-31", "2026-08-01"),
            )
            for row in [
                (goal_id, "2026-08-10", 5000.0, "Saved from pocket money"),
                (goal_id, "2026-08-25", 4000.0, "Freelance payment"),
                (goal_id, "2026-09-10", 6000.0, "Monthly saving"),
            ]:
                execute(
                    "INSERT INTO savings_history (goal_id, date, amount, note) VALUES (?, ?, ?, ?)", row
                )

            execute(
                "INSERT INTO savings_goals (name, target_amount, target_date, created_on) VALUES (?, ?, ?, ?)",
                ("Goa Trip Fund", 15000.0, "2026-11-15", "2026-09-01"),
            )

            demo_inv = [
                ("Nifty 50 Index Fund", "Mutual Fund / SIP", 12000.0, 13450.0, "2026-04-15", "Monthly SIP via Groww"),
                ("SBI Fixed Deposit", "Fixed Deposit", 20000.0, 21100.0, "2026-03-01", "1 year tenure at 7.1%"),
                ("Tata Motors", "Stocks / Equity", 6000.0, 5400.0, "2026-06-20", "Long term hold"),
                ("Digital Gold", "Gold", 4000.0, 4380.0, "2026-05-10", "Accumulating monthly"),
            ]
            for row in demo_inv:
                execute(
                    "INSERT INTO investments (name, type, amount, current_value, date, notes) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    row,
                )

            execute(
                "INSERT INTO investment_goals (name, target_amount, target_date, created_on) VALUES (?, ?, ?, ?)",
                ("1 Lakh Portfolio", 100000.0, "2027-03-31", "2026-09-01"),
            )

            st.success("Demo data loaded. Open the Dashboard to see it.")
            st.rerun()

    st.divider()
    st.markdown("#### Export Data")
    exp_df = query_df("SELECT date, category, amount, description FROM expenses ORDER BY date")
    if exp_df.empty:
        st.caption("Nothing to export yet.")
    else:
        exp_df.columns = ["Date", "Category", "Amount", "Description"]
        st.download_button(
            "Download expenses as CSV",
            data=exp_df.to_csv(index=False).encode("utf-8"),
            file_name="expenses_export.csv",
            mime="text/csv",
        )

    st.divider()
    st.markdown("#### Reset Database")
    st.caption("This permanently deletes every expense, fund entry, goal and investment.")
    confirm = st.checkbox("I understand this cannot be undone")
    if st.button("Reset All Data", disabled=not confirm):
        for table in [
            "savings_history",
            "savings_goals",
            "emergency_fund",
            "investments",
            "investment_goals",
            "expenses",
        ]:
            execute(f"DELETE FROM {table}")
        st.success("All data cleared.")
        st.rerun()

    st.divider()
    st.markdown("#### System Information")
    st.write(
        {
            "Database file": DB_FILE,
            "Expense records": int(scalar("SELECT COUNT(*) FROM expenses", default=0)),
            "Emergency fund entries": int(scalar("SELECT COUNT(*) FROM emergency_fund", default=0)),
            "Savings goals": int(scalar("SELECT COUNT(*) FROM savings_goals", default=0)),
            "Investments": int(scalar("SELECT COUNT(*) FROM investments", default=0)),
            "ML categories": len(CATEGORIES),
            "Training samples": len(train_data),
        }
    )
