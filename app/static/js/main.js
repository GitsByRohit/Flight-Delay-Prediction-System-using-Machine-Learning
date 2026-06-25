/* ======================================
   PAGE FADE-IN EFFECT
====================================== */

document.addEventListener("DOMContentLoaded", () => {
    document.body.classList.add("loaded");
});


/* ======================================
   PREDICTION BUTTON LOADING STATE
====================================== */

const form = document.querySelector("form");

if(form){

    form.addEventListener("submit", () => {

        const btn = document.getElementById("predict-btn");

        if(btn){

            btn.innerHTML = "⏳ Predicting...";
            btn.disabled = true;

        }

    });

}


/* ======================================
   CONFIDENCE BAR ANIMATION
====================================== */

const progressBar = document.querySelector(".progress-bar");

if(progressBar){

    const finalWidth = progressBar.style.width;

    progressBar.style.width = "0%";

    setTimeout(() => {

        progressBar.style.transition = "width 1.5s ease";
        progressBar.style.width = finalWidth;

    }, 300);

}