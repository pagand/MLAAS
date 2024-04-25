function createCanvas(temp_key, type, query, log_content_py = {}) {
    const html2canvas = window.html2canvas;
    const streamDocument = window.parent.document;
    const clickedBtn = streamDocument.getElementById(`${temp_key}`);
    const closestDataStale = clickedBtn.closest("[data-stale]");
    let targetDataParent = closestDataStale.previousElementSibling;
    let title = null;

    while (targetDataParent) {
        if (targetDataParent.querySelector(".stTextInput")) {
            const textInput = targetDataParent.querySelector("input[type=text]");
            textInput.setAttribute("id", `${temp_key}titleTextInput`);
            title = textInput.value;
            textInput.value = "";
        } else if (targetDataParent.querySelector(".stMarkdown")) {
            targetDataParent.setAttribute("id", `${temp_key}imageTitleLabel`);
        } else if (targetDataParent.querySelector(`${query}`)) break;
        targetDataParent = targetDataParent.previousElementSibling;
    }
    const targetData = targetDataParent.querySelector(`${query}`);
    const targetHeight = targetData.offsetHeight;
    const targetWidth = targetData.offsetWidth;

    setTimeout(() => {
        html2canvas(targetData, {
            allowTaint: true,
            width: targetWidth,
            height: targetHeight,
            scale: 1,
        }).then(function (canvas) {
            const imageDataURL = canvas.toDataURL("image/png");

            const log_content = {
                ...log_content_py,
                imageDataURL,
                title,
            };

            window.top.postMessage(JSON.stringify(log_content), "https://da-tu.ca")

            clickedBtn.innerText = "Screenshot saved";
                clickedBtn.disabled = true;
                streamDocument.getElementById(`${temp_key}imageTitleLabel`).style.display = "none";
                streamDocument.getElementById(`${temp_key}titleTextInput`).style.display = "none";
                if (type === "plotly_chart") {
                    const saveAnotherScreenshotBtn = streamDocument.createElement("button");
                    saveAnotherScreenshotBtn.innerText = "Save another screenshot?";
                    saveAnotherScreenshotBtn.className = clickedBtn.className;
                    saveAnotherScreenshotBtn.id = `${temp_key}saveAnotherBtn`;

                    saveAnotherScreenshotBtn.style.marginLeft = "1rem";
                    clickedBtn.parentNode.appendChild(saveAnotherScreenshotBtn);

                    // Setup event listener for save another screenshot button
                    saveAnotherScreenshotBtn.addEventListener("click", function () {
                        clickedBtn.innerText = "Save to notes";
                        clickedBtn.disabled = false;
                        streamDocument.getElementById(`${temp_key}imageTitleLabel`).style.display =
                            "block";
                        streamDocument.getElementById(`${temp_key}titleTextInput`).value = "";
                        streamDocument.getElementById(`${temp_key}titleTextInput`).style.display =
                            "block";
                        this.parentNode.removeChild(this);
                    });
                }
        });
    }, 1200);
}
