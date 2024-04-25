function sendLog(log_content = {}) {
    window.top.postMessage(JSON.stringify(log_content), "https://da-tu.ca");
}
