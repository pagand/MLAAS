function updateButtonLabel(temp_key, type) {
    setTimeout(() => {
        const streamDocument = window.parent.document;
        const allBtns = streamDocument.querySelectorAll(".stButton > button");

        const saveImageBtn = [...allBtns].find(
            (btn) => btn.innerText === temp_key || btn.id === temp_key
        );

        if (saveImageBtn) {
            saveImageBtn.setAttribute("id", temp_key);
            saveImageBtn.innerText = "Save image to notes";

            if (type === "dataframe") {
                let targetDataParent = saveImageBtn.closest("[data-stale]").previousElementSibling;

                // Short delay to allow time for dataframe to load before searching for it
                while (targetDataParent) {
                    if (targetDataParent.querySelector(".stTextInput")) {
                        const textInput = targetDataParent.querySelector("input[type=text]");
                        textInput.setAttribute("id", `${temp_key}titleTextInput`);
                        textInput.value = "";
                    } else if (targetDataParent.querySelector(".stMarkdown")) {
                        targetDataParent.setAttribute("id", `${temp_key}imageTitleLabel`);
                    } else if (targetDataParent.querySelector(".stDataFrame")) break;
                    targetDataParent = targetDataParent.previousElementSibling;
                }

                const frameAdjustContainer = targetDataParent.querySelector(
                    '.stDataFrame [data-testid="stDataFrameResizable"]'
                );
                frameAdjustContainer.style.maxWidth = "100%";
            }
        }

        const allBtnIDs = [...allBtns].map((btn) => btn.id);
        const saveImageAgainBtnIDs = allBtnIDs.filter((id) => id.search("saveAnotherBtn") !== -1);

        for (const btnID of saveImageAgainBtnIDs) {
            const saveImageAgainBtn = streamDocument.getElementById(btnID);
            const new_temp_key = btnID.replace(/saveAnotherBtn/, "");
            const newSaveImageBtn = streamDocument.getElementById(new_temp_key);
            const imageTitleLabel = streamDocument.getElementById(`${new_temp_key}imageTitleLabel`);
            const titleTextInput = streamDocument.getElementById(`${new_temp_key}titleTextInput`);

            saveImageAgainBtn.addEventListener("click", function () {
                newSaveImageBtn.innerText = "Save to notes";
                newSaveImageBtn.disabled = false;
                imageTitleLabel.style.display = "block";
                titleTextInput.value = "";
                titleTextInput.style.display = "block";
                this.parentNode.removeChild(this);
            });
        }
    }, 1000);
}
