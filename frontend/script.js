document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const previewContainer = document.getElementById('preview-container');
    const imagePreview = document.getElementById('image-preview');
    const removeBtn = document.getElementById('remove-btn');
    const analyzeBtn = document.getElementById('analyze-btn');
    const resultSection = document.getElementById('result-section');
    const verdict = document.getElementById('verdict');
    const confidenceText = document.getElementById('confidence-text');
    const progressFill = document.getElementById('progress-fill');
    const scannerBar = document.getElementById('scanner-bar');
    const logWindow = document.getElementById('log-window');

    let selectedFile = null;

    // Utility: Append to logs
    function appendLog(message, isError=false, isSuccess=false) {
        let prefix = "> ";
        let colorClass = "";
        if(isError) colorClass = "text-neon-red";
        if(isSuccess) colorClass = "text-neon-cyan";
        
        const timestamp = new Date().toISOString().split('T')[1].substring(0,8);
        const logLine = document.createElement('p');
        logLine.className = `log-line ${colorClass}`;
        logLine.textContent = `[${timestamp}] ${prefix}${message}`;
        logWindow.appendChild(logLine);
        logWindow.scrollTop = logWindow.scrollHeight;
    }

    // Drag and Drop Mapping
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eName => {
        dropZone.addEventListener(eName, e => { e.preventDefault(); e.stopPropagation(); });
    });

    ['dragenter', 'dragover'].forEach(eName => dropZone.addEventListener(eName, () => dropZone.classList.add('dragover')));
    ['dragleave', 'drop'].forEach(eName => dropZone.addEventListener(eName, () => dropZone.classList.remove('dragover')));

    dropZone.addEventListener('drop', e => handleFiles(e.dataTransfer.files));
    dropZone.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', e => handleFiles(e.target.files));

    function handleFiles(files) {
        if (files.length > 0) {
            const file = files[0];
            if (file.type.startsWith('image/')) {
                selectedFile = file;
                appendLog(`DATA AQUIRED: ${file.name} (${(file.size/1024).toFixed(1)} KB)`);
                showPreview(file);
                analyzeBtn.disabled = false;
                resultSection.classList.add('hidden');
                previewContainer.classList.remove('scanned');
            } else {
                appendLog("ERR: INVALID DATA FORMAT. REQUIRES IMAGE TENSOR.", true);
            }
        }
    }

    function showPreview(file) {
        const reader = new FileReader();
        reader.readAsDataURL(file);
        reader.onloadend = () => {
            imagePreview.src = reader.result;
            dropZone.classList.add('hidden');
            previewContainer.classList.remove('hidden');
        };
    }

    removeBtn.addEventListener('click', () => {
        selectedFile = null;
        fileInput.value = '';
        dropZone.classList.remove('hidden');
        previewContainer.classList.add('hidden');
        imagePreview.src = '';
        analyzeBtn.disabled = true;
        resultSection.classList.add('hidden');
        appendLog("TARGET DATA FLUSHED FROM MEMORY.");
    });

    // Neural Scan Button Trigger
    analyzeBtn.addEventListener('click', async () => {
        if (!selectedFile) return;

        // Sequence UI updates
        analyzeBtn.disabled = true;
        analyzeBtn.querySelector('.btn-text').textContent = "SCANNING...";
        resultSection.classList.add('hidden');
        scannerBar.classList.add('scanning');
        
        appendLog(`INITIATING UPLINK TO ENSEMBLE AI CORE...`);

        const formData = new FormData();
        formData.append('file', selectedFile);

        try {
            // Fake delay for dramatic matrix effect
            appendLog(`EXTRACTING SPATIAL FEATURES...`);
            await new Promise(r => setTimeout(r, 600));
            appendLog(`MAPPING TENSORS TO XCEPTION WEIGHTS...`);
            await new Promise(r => setTimeout(r, 600));

            const response = await fetch('/api/predict', {
                method: 'POST',
                body: formData
            });

            appendLog(`SERVER RESPONDED: STATUS ${response.status}`, !response.ok);

            const data = await response.json();
            if (!response.ok) throw new Error(data.detail || "Prediction failed");

            appendLog("DECODING SIGMOID CLASSIFICATION...", false, true);
            await new Promise(r => setTimeout(r, 400));
            
            displayResults(data.prediction, data.confidence);

        } catch (error) {
            appendLog(`CRITICAL FAILURE: ${error.message}`, true);
        } finally {
            analyzeBtn.disabled = false;
            analyzeBtn.querySelector('.btn-text').textContent = "EXECUTE NEURAL SCAN";
            scannerBar.classList.remove('scanning');
        }
    });

    function displayResults(label, confidence) {
        previewContainer.classList.add('scanned');
        resultSection.classList.remove('hidden');
        verdict.textContent = label;
        verdict.setAttribute('data-text', label);
        
        verdict.className = 'glitch-verdict'; // reset
        if (label === 'REAL') {
            verdict.classList.add('verdict-real');
            progressFill.style.backgroundColor = 'var(--neon-green)';
            appendLog("TARGET CLASSIFIED AS GENUINE.", false, true);
        } else {
            verdict.classList.add('verdict-fake');
            progressFill.style.backgroundColor = 'var(--neon-red)';
            appendLog("WARNING: SYNTHETIC MANIPULATION DETECTED.", true);
        }

        const percentage = (confidence * 100).toFixed(1);
        confidenceText.textContent = `${percentage}%`;
        
        setTimeout(() => {
            progressFill.style.width = `${percentage}%`;
        }, 100);
    }
});
