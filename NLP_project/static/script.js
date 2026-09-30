const icons = {
    sport: "⚽",
    business: "💼",
    politics: "🏛",
    tech: "💻",
    entertainment: "🎬"
};

const fileInput = document.getElementById("fileInput");
const dropZone = document.getElementById("dropZone");

fileInput.addEventListener("change", () => {
    if (fileInput.files.length) {
        setSelectedFile(fileInput.files[0]);
    }
});

["dragenter", "dragover"].forEach(eventName => {
    dropZone.addEventListener(eventName, e => {
        e.preventDefault();
        dropZone.classList.add("dragover");
    });
});

["dragleave", "drop"].forEach(eventName => {
    dropZone.addEventListener(eventName, e => {
        e.preventDefault();
        dropZone.classList.remove("dragover");
    });
});

dropZone.addEventListener("drop", e => {
    const files = e.dataTransfer.files;

    if (files.length) {
        setSelectedFile(files[0]);
    }
});

function setSelectedFile(file) {
    const allowed = [
        ".pdf", ".txt", ".csv", ".json",
        ".docx", ".xlsx", ".zip"
    ];

    const name = file.name.toLowerCase();
    const valid = allowed.some(ext => name.endsWith(ext));

    if (!valid) {
        alert("Unsupported file type. Please upload PDF, TXT, CSV, JSON, DOCX, XLSX or ZIP.");
        return;
    }

    const dataTransfer = new DataTransfer();
    dataTransfer.items.add(file);
    fileInput.files = dataTransfer.files;

    const extension = file.name.split(".").pop().toUpperCase();

    document.getElementById("fileIcon").textContent = extension;
    document.getElementById("fileName").textContent = file.name;
    document.getElementById("fileSize").textContent =
        formatBytes(file.size);

    document.getElementById("fileInfo").classList.remove("hidden");
    document.getElementById("classifyBtn").disabled = false;
    document.getElementById("statusText").textContent =
        "File ready for analysis.";
}

function formatBytes(bytes) {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024)
        return `${(bytes / 1024).toFixed(1)} KB`;

    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function removeFile() {
    fileInput.value = "";

    document.getElementById("fileInfo").classList.add("hidden");
    document.getElementById("classifyBtn").disabled = true;
    document.getElementById("statusText").textContent = "";
}

async function analyzeFile() {
    if (!fileInput.files.length) {
        alert("Please select a file first.");
        return;
    }

    const button = document.getElementById("classifyBtn");
    const status = document.getElementById("statusText");

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);

    button.disabled = true;
    button.innerHTML =
        `<span>⟳</span> Extracting & analyzing...`;

    status.textContent =
        "Reading the file and running the NLP model...";

    try {
        const response = await fetch("/predict-file", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || "Analysis failed.");
        }

        showResult(data);

        status.textContent =
            "Analysis completed successfully.";

    } catch (error) {
        alert(error.message);
        status.textContent = "";
    } finally {
        button.disabled = false;
        button.innerHTML =
            `<span>✦</span> Analyze File <span class="arrow">→</span>`;

        if (fileInput.files.length) {
            button.disabled = false;
        }
    }
}

function showResult(data) {
    document.getElementById("emptyState").classList.add("hidden");
    document.getElementById("resultState").classList.remove("hidden");

    const category = data.category;

    document.getElementById("categoryIcon").textContent =
        icons[category.toLowerCase()] || "✦";

    document.getElementById("categoryName").textContent =
        category;

    document.getElementById("confidenceValue").textContent =
        `${data.confidence}%`;

    document.getElementById("confidenceLevel").textContent =
        data.confidence_level;

    document.getElementById("confidenceBar").style.width =
        `${data.confidence}%`;

    document.getElementById("resultFileIcon").textContent =
        data.file_type.toUpperCase();

    document.getElementById("resultFileName").textContent =
        data.filename;

    document.getElementById("resultFileType").textContent =
        data.file_type;

    document.getElementById("pageCount").textContent =
        data.pages || data.items || 1;

    document.getElementById("wordCount").textContent =
        data.word_count.toLocaleString();

    document.getElementById("readingTime").textContent =
        `${data.reading_time} min`;

    document.getElementById("extractedText").textContent =
        data.extracted_preview +
        (data.extracted_preview.length >= 3500
            ? "\n\n…preview truncated…"
            : "");

    renderProbabilities(data.probabilities);
    renderKeywords(data.keywords);

    const ignoredBox =
        document.getElementById("ignoredBox");

    if (data.ignored_files && data.ignored_files.length) {
        ignoredBox.classList.remove("hidden");

        ignoredBox.textContent =
            `ZIP notice: ${data.ignored_files.length} file(s) inside the archive `
            + `could not be analyzed because their format is unsupported or unreadable.`;
    } else {
        ignoredBox.classList.add("hidden");
    }

    document.getElementById("resultState").scrollIntoView({
        behavior: "smooth",
        block: "nearest"
    });
}

function renderProbabilities(items) {
    const container =
        document.getElementById("probabilities");

    container.innerHTML = "";

    items.forEach(item => {
        const row = document.createElement("div");

        row.className = "prob-row";

        row.innerHTML = `
            <div class="prob-header">
                <span>${item.category}</span>
                <span>${item.probability}%</span>
            </div>

            <div class="prob-track">
                <div class="prob-fill"
                     style="width:${item.probability}%">
                </div>
            </div>
        `;

        container.appendChild(row);
    });
}

function renderKeywords(items) {
    const container =
        document.getElementById("keywords");

    container.innerHTML = "";

    if (!items.length) {
        container.innerHTML =
            "<span>No strong keywords found</span>";

        return;
    }

    items.forEach(keyword => {
        const tag = document.createElement("span");

        tag.textContent = keyword;

        container.appendChild(tag);
    });
}
