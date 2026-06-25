from flask import Blueprint, render_template, request, redirect, url_for, session, flash, send_file
from app.services.model_service import predict_delay
from app.utils.form_preprocessor import preprocess_input
from app.utils.pdf_generator import generate_prediction_pdf

prediction_bp = Blueprint("prediction", __name__)


# LOGIN PROTECTION
def login_required(func):
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            flash("Please login first.", "warning")
            return redirect(url_for("auth.login"))
        return func(*args, **kwargs)
    wrapper.__name__ = func.__name__
    return wrapper


# AIRPORT LIST
airport_list = [
'IND','ISP','JAN','JAX','LAS','LAX','LBB','LIT','MAF','MCI','MCO','MDW','MHT','MSY','OAK','OKC','OMA','ONT',
'ORF','PBI','PDX','PHL','PHX','PIT','PVD','RDU','RNO','RSW','SAN','SAT','SDF','SEA','SFO','SJC','SLC','SMF',
'SNA','STL','TPA','TUL','TUS','ABQ','AMA','AUS','BHM','BNA','BOI','BUF','BUR','BWI','CMH','CRP','DAL','DEN',
'ELP','FLL','GEG','HOU','HRL','IAD','ALB','BDL','DTW','CLE','ORD','ATL','CVG','MKE','MSP','EWR','SUN','SGU',
'MSO','BZN','GTF','BIL','PSP','HDN','IAH','DFW','ASE','JAC','SBP','FAT','EUG','MOD','LEX','FSD','BTV','ROA',
'MEM','FAR','XNA','COS','GUC','AZO','TVC','CRW','TYS','SAV','ICT','GJT','PIA','SGF','MSN','CID','MLI','DRO',
'CHS','DSM','ATW','GRB','FWA','DAY','LNK','FCA','IDA','HSV','CWA','MFR','PSC','SYR','SBA','RAP','YUM','RDM',
'LGB','MTJ','RDD','CLD','SBN','HPN','SPI','MBS','LAN','TWF','MRY','SMX','ACV','BFL','CEC','CIC','PMD','EKO',
'IYK','OXR','IPL','PIH','CPR','BTM','HLN','BLI','CAK','RFD','COD','SLE','LWS','GRR','AVP','ABE','BIS','GSP',
'CDC','BMI','YKM','CLT','HNL','KOA','OGG','JFK','LIH','MDT','LGA','RIC','BOS','EGE','ROC','GSO','DCA','SJU',
'STT','MIA','ANC','MYR','STX','ILM','VPS','SRQ','PNS','DAB','CAE','GPT','MLB','PHF','MFE','SHV','MGM','PFN',
'CHA','FAY','AGS','MOB','BTR','BGR','GNV','ABY','DHN','AVL','EVV','FNT','TRI','OAJ','AEX','SWF','EWN','MEI',
'PWM','GRK','GTR','LFT','LYH','HHH','EYW','VLD','CSG','MLU','TLH','ACY','FSM','MCN','CHO','TOL','FLO','BQK',
'SCE','ITO','LAW','SPS','ABI','CLL','TYR','GGG','ACT','SJT','TXK','LRD','CMI','ROW','RST','MQT','LSE','DBQ',
'KTN','JNU','SIT','PSG','CDV','YAK','BET','BRW','SCC','FAI','ADQ','WRG','OME','OTZ','ADK','PSE','BQN','MKG',
'DLG','AKN','LWB','WYS'
]


# DASHBOARD
@prediction_bp.route("/")
@login_required
def dashboard():

    airlines = [
        "Alaska Airlines Inc.",
        "American Airlines Inc.",
        "American Eagle Airlines Inc.",
        "Atlantic Southeast Airlines",
        "Delta Air Lines Inc.",
        "Frontier Airlines Inc.",
        "Hawaiian Airlines Inc.",
        "JetBlue Airways",
        "Skywest Airlines Inc.",
        "Southwest Airlines Co.",
        "US Airways Inc.",
        "United Air Lines Inc."
    ]

    months = list(range(1, 13))

    return render_template(
        "dashboard.html",
        username=session.get("user_name"),
        airlines=airlines,
        months=months,
        origins=airport_list,
        destinations=airport_list
    )


# PREDICTION
@prediction_bp.route("/predict", methods=["POST"])
@login_required
def predict():

    try:

        form_data = request.form.to_dict()

        model_input = preprocess_input(form_data)

        prediction, confidence = predict_delay(model_input)

        # CONFIDENCE FIX (no extra multiplication)
        confidence_percent = round(confidence, 2)

        if prediction == 1:
            result = "Flight will be Delayed ✈️"
            alert_class = "danger"
            message = "⚠️ The flight is likely to be delayed."
            prediction_label = "Delayed"
        else:
            result = "Flight will be On-Time ✅"
            alert_class = "success"
            message = "✅ The flight is expected to be on time."
            prediction_label = "On-Time"


        # STORE REPORT DATA FOR PDF
        session["prediction_report"] = {

            "airline": request.form.get("airline"),

            "origin_airport": request.form.get("origin_airport"),

            "destination_airport": request.form.get("destination_airport"),

            "month": request.form.get("month"),

            "day": request.form.get("day"),

            "day_of_week": request.form.get("day_of_week"),

            "departure_hour": request.form.get("departure_hour"),

            "distance": request.form.get("distance"),

            "prediction": prediction_label,

            "confidence": confidence_percent,

            "threshold": 40,

            "model_name": "Support Vector Machine",

            "accuracy": "80.8%",

            "f1_score": "0.89",

            "dataset_size": "484,000 flights"
        }


        return render_template(
            "result.html",
            prediction_text=result,
            confidence=confidence_percent,
            alert_class=alert_class,
            message=message
        )


    except Exception as e:

        flash(f"Error: {str(e)}", "danger")

        return redirect(url_for("prediction.dashboard"))


# DOWNLOAD PDF
@prediction_bp.route("/download-report")
@login_required
def download_report():

    try:

        report_data = session.get("prediction_report")

        if not report_data:
            flash("No prediction report available.", "warning")
            return redirect(url_for("prediction.dashboard"))

        pdf_buffer = generate_prediction_pdf(report_data)

        return send_file(
            pdf_buffer,
            as_attachment=True,
            download_name="flight_delay_prediction_report.pdf",
            mimetype="application/pdf"
        )

    except Exception as e:

        flash(f"Error generating report: {str(e)}", "danger")

        return redirect(url_for("prediction.dashboard"))