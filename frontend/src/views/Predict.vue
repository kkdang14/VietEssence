<template>
    <div class="main">
        <div class="main-container">
            <!-- Classify Container - Left Side -->
            <div class="classify-container">
                <div class="section-header">
                    <h3><i class="fa-solid fa-image"></i> Phân loại ảnh</h3>
                </div>
                
                <!-- Upload Mode Selector -->
                <div class="upload-mode-selector">
                    <label>
                        <input type="radio" v-model="uploadMode" value="single" />
                        Tải ảnh đơn
                    </label>
                    <label>
                        <input type="radio" v-model="uploadMode" value="multiple" />
                        Tải nhiều ảnh
                    </label>
                    <label>
                        <input type="radio" v-model="uploadMode" value="zip" />
                        Tải folder ZIP
                    </label>
                </div>

                <!-- Upload Section -->
                <div class="upload-section">
                    <div class="upload-area" @click="triggerFileInput">
                        <div class="upload-placeholder" v-if="!isUploading">
                            <i class="fa-solid fa-cloud-arrow-up"></i>
                            <p v-if="uploadMode === 'single'">Nhấp để tải một ảnh</p>
                            <p v-else-if="uploadMode === 'multiple'">Nhấp để tải nhiều ảnh</p>
                            <p v-else>Nhấp để tải folder ZIP chứa ảnh</p>
                            <span>hoặc kéo thả {{ uploadMode === 'zip' ? 'folder ZIP' : 'ảnh' }} vào đây</span>
                        </div>
                        <div class="upload-loading" v-else>
                            <i class="fa-solid fa-spinner fa-spin"></i>
                            <p>Đang xử lý...</p>
                        </div>
                    </div>
                    <input type="file" id="fileInput" 
                        :accept="uploadMode === 'zip' ? '.zip' : 'image/*'" 
                        @change="onFilesSelected" 
                        ref="fileInput" 
                        :multiple="uploadMode !== 'single'" />
                </div>

                <!-- Action Buttons -->
                <div class="action-buttons">
                    <button class="btn-predict" @click="predictImages" :disabled="uploadedImages.length === 0 || isProcessing">
                        <i class="fa-solid fa-magic-wand-sparkles"></i>
                        {{ isProcessing ? 'Đang xử lý...' : 'Phân loại' }}
                    </button>
                    <button class="btn-clear" @click="clearUploadedImages" :disabled="uploadedImages.length === 0 || isProcessing">
                        <i class="fa-solid fa-trash-can"></i>
                        Xóa tất cả
                    </button>
                </div>

                <!-- Uploaded Images List -->
                <div class="uploaded-images-list" v-if="uploadedImages.length > 0">
                    <div class="section-header">
                        <h4><i class="fa-solid fa-images"></i> Danh sách ảnh đã chọn ({{ uploadedImages.length }})</h4>
                    </div>
                    <div class="image-box">
                        <div v-for="(image, index) in uploadedImages" :key="index" class="image-box__item">
                            <img :src="image.url" :alt="image.name" class="image-box__thumbnail" @error="handleImageError('uploaded', index)" />
                            <div class="image-box__info">
                                <span class="image-box__name">{{ image.name }}</span>
                            </div>
                            <div class="image-box__actions">
                                <button class="btn-delete-image" @click="removeUploadedImage(index)">
                                    <i class="fa-solid fa-trash"></i>
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Store Container - Right Side -->
            <div class="store-container">
                <div class="section-header">
                    <h3><i class="fa-solid fa-folder"></i> Thư viện ảnh phân loại</h3>
                    <div class="store-actions">
                        <button class="btn-download-all" @click="downloadAllImages" :disabled="!hasStoredImages">
                            <i class="fa-solid fa-download"></i>
                            Tải tất cả
                        </button>
                        <button class="btn-clear-session" @click="clearSessionData" :disabled="!hasStoredImages">
                            <i class="fa-solid fa-trash-can"></i>
                            Xóa phiên
                        </button>
                    </div>
                </div>

                <!-- Folder List -->
                <div class="folder-list" v-if="Object.keys(storedImages).length > 0">
                    <div v-for="(images, className) in storedImages" :key="className" class="folder-item">
                        <div class="folder-header" @click="toggleFolder(className)">
                            <div class="folder-info">
                                <i class="fa-solid" :class="expandedFolders[className] ? 'fa-folder-open' : 'fa-folder'"></i>
                                <span class="folder-name">{{ className }}</span>
                                <span class="image-count">({{ images.length }})</span>
                            </div>
                            <div class="folder-actions">
                                <button class="btn-download-folder" @click.stop="downloadFolder(className)">
                                    <i class="fa-solid fa-download"></i>
                                </button>
                                <button class="btn-delete-folder" @click.stop="deleteFolder(className)">
                                    <i class="fa-solid fa-trash"></i>
                                </button>
                            </div>
                        </div>
                        
                        <div v-if="expandedFolders[className]" class="folder-content">
                            <div class="image-grid">
                                <div v-for="(image, index) in images" :key="index" class="image-item">
                                    <img :src="image.url" :alt="image.name" class="thumbnail" @error="handleImageError(className, index)" />
                                    <div class="image-info">
                                        <span class="image-name">{{ image.name }}</span>
                                        <span class="image-confidence">{{ image.confidence }}%</span>
                                    </div>
                                    <!-- <div class="image-actions">
                                        <button class="btn-download-image" @click="downloadImage(image)">
                                            <i class="fa-solid fa-download"></i>
                                        </button>
                                        <button class="btn-delete-image" @click="deleteImage(className, index)">
                                            <i class="fa-solid fa-trash"></i>
                                        </button>
                                    </div> -->
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Empty State -->
                <div v-else class="empty-state">
                    <i class="fa-solid fa-images"></i>
                    <p>Chưa có ảnh nào được lưu</p>
                    <span>Hãy phân loại và lưu ảnh vào thư viện</span>
                </div>
            </div>
        </div>
    </div>
</template>

<script>
import PredictService from "../services/predict.service";
import AppHeader from "@/components/AppHeader.vue";
import JSZip from "jszip";

export default {
    name: "ImageClassifyPage",
    components: {
        AppHeader
    },
    data() {
        return {
            uploadedImages: [], 
            currentImageIndex: -1, 
            isProcessing: false,
            isUploading: false, 
            predictionResult: null,
            storedImages: {},
            expandedFolders: {},
            imageBlobs: {},
            uploadMode: 'single', // Default mode: 'single', 'multiple', 'zip'
            db: null // IndexedDB instance
        };
    },
    computed: {
        hasStoredImages() {
            return Object.keys(this.storedImages).length > 0;
        },
        hasUploadedImages() {
            return this.uploadedImages.length > 0;
        },
    },
    methods: {
        // Initialize IndexedDB
        async initIndexedDB() {
            return new Promise((resolve, reject) => {
                const request = indexedDB.open('ImageClassifierDB', 1);

                request.onupgradeneeded = (event) => {
                    const db = event.target.result;
                    if (!db.objectStoreNames.contains('storedImages')) {
                        db.createObjectStore('storedImages', { keyPath: 'key' });
                    }
                    if (!db.objectStoreNames.contains('expandedFolders')) {
                        db.createObjectStore('expandedFolders', { keyPath: 'key' });
                    }
                };

                request.onsuccess = (event) => {
                    this.db = event.target.result;
                    resolve();
                };

                request.onerror = (event) => {
                    console.error('Error opening IndexedDB:', event.target.error);
                    reject(event.target.error);
                };
            });
        },

        // Save data to IndexedDB
        async saveToIndexedDB() {
            if (!this.db) return;

            try {
                const tx = this.db.transaction(['storedImages', 'expandedFolders'], 'readwrite');
                const storedImagesStore = tx.objectStore('storedImages');
                const expandedFoldersStore = tx.objectStore('expandedFolders');

                storedImagesStore.put({ key: 'storedImages', value: this.storedImages });
                expandedFoldersStore.put({ key: 'expandedFolders', value: this.expandedFolders });

                await new Promise((resolve, reject) => {
                    tx.oncomplete = () => resolve();
                    tx.onerror = () => reject(tx.error);
                });
            } catch (error) {
                console.error('Error saving to IndexedDB:', error);
                if (error.name === 'QuotaExceededError') {
                    alert('Bộ nhớ IndexedDB đã đầy! Vui lòng xóa một số ảnh để tiếp tục.');
                }
            }
        },

        // Load data from IndexedDB
        async loadFromIndexedDB() {
            if (!this.db) return;

            try {
                const tx = this.db.transaction(['storedImages', 'expandedFolders'], 'readonly');
                const storedImagesStore = tx.objectStore('storedImages');
                const expandedFoldersStore = tx.objectStore('expandedFolders');

                const storedImagesRequest = storedImagesStore.get('storedImages');
                const expandedFoldersRequest = expandedFoldersStore.get('expandedFolders');

                const storedImagesResult = await new Promise((resolve, reject) => {
                    storedImagesRequest.onsuccess = () => resolve(storedImagesRequest.result);
                    storedImagesRequest.onerror = () => reject(storedImagesRequest.error);
                });

                const expandedFoldersResult = await new Promise((resolve, reject) => {
                    expandedFoldersRequest.onsuccess = () => resolve(expandedFoldersRequest.result);
                    expandedFoldersRequest.onerror = () => reject(expandedFoldersRequest.error);
                });

                if (storedImagesResult) {
                    this.storedImages = storedImagesResult.value || {};
                }
                if (expandedFoldersResult) {
                    this.expandedFolders = expandedFoldersResult.value || {};
                }
            } catch (error) {
                console.error('Error loading from IndexedDB:', error);
                this.storedImages = {};
                this.expandedFolders = {};
            }
        },

        // Clear IndexedDB data
        async clearSessionData() {
            if (confirm("Bạn có chắc chắn muốn xóa tất cả dữ liệu phiên làm việc?")) {
                try {
                    const tx = this.db.transaction(['storedImages', 'expandedFolders'], 'readwrite');
                    tx.objectStore('storedImages').clear();
                    tx.objectStore('expandedFolders').clear();

                    await new Promise((resolve, reject) => {
                        tx.oncomplete = () => resolve();
                        tx.onerror = () => reject(tx.error);
                    });

                    this.storedImages = {};
                    this.expandedFolders = {};
                    this.imageBlobs = {};
                    alert("Đã xóa tất cả dữ liệu phiên!");
                } catch (error) {
                    console.error('Error clearing IndexedDB:', error);
                    alert('Có lỗi khi xóa dữ liệu phiên!');
                }
            }
        },

        triggerFileInput() {
            this.$refs.fileInput.click();
        },

        async onFilesSelected(event) {
            const files = Array.from(event.target.files);
            if (files.length === 0) return;

            this.isUploading = true;
            this.predictionResult = null;
            this.currentImageIndex = -1;

            try {
                if (this.uploadMode === 'single' && files.length > 1) {
                    alert("Vui lòng chỉ chọn một ảnh cho chế độ tải ảnh đơn!");
                    return;
                }

                if (this.uploadMode === 'zip') {
                    if (files.length > 1 || !files[0].name.endsWith('.zip')) {
                        alert("Vui lòng chỉ chọn một file ZIP!");
                        return;
                    }
                    await this.handleZipFile(files[0]);
                } else {
                    for (const file of files) {
                        if (!file.type.startsWith("image/") || !file.name.match(/\.(jpg|jpeg|png|gif)$/i)) {
                            alert(`File ${file.name} không phải ảnh hợp lệ (jpg, jpeg, png, gif)!`);
                            continue;
                        }
                        await this.handleSingleImage(file);
                    }
                }

                if (this.uploadedImages.length === 0 && this.uploadMode !== 'zip') {
                    alert("Không có ảnh hợp lệ được chọn!");
                }
            } catch (error) {
                console.error("Error processing files:", error);
                alert("Có lỗi khi xử lý file. Vui lòng thử lại!");
            } finally {
                this.isUploading = false;
                this.$refs.fileInput.value = '';
            }
        },

        async handleSingleImage(file) {
            const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10MB
            if (file.size > MAX_FILE_SIZE) {
                alert(`File ${file.name} quá lớn (giới hạn 10MB).`);
                return;
            }

            if (this.uploadedImages.some(img => img.name === file.name)) {
                console.warn(`Duplicate file detected: ${file.name}`);
                return;
            }

            // Validate image by loading it
            const img = new Image();
            const isValid = await new Promise(resolve => {
                img.onload = () => resolve(true);
                img.onerror = () => resolve(false);
                img.src = URL.createObjectURL(file);
            });

            if (!isValid) {
                alert(`File ${file.name} không phải ảnh hợp lệ hoặc bị hỏng!`);
                return;
            }

            const imageData = {
                file: file,
                url: URL.createObjectURL(file),
                name: file.name,
            };
            this.uploadedImages.push(imageData);
        },

        async handleZipFile(zipFile) {
            const MAX_ZIP_SIZE = 100 * 1024 * 1024; // 100MB
            if (zipFile.size > MAX_ZIP_SIZE) {
                alert(`File ZIP ${zipFile.name} quá lớn (giới hạn 100MB).`);
                return;
            }

            const zip = new JSZip();
            try {
                const zipContent = await zip.loadAsync(zipFile);
                const imagePromises = [];

                console.log(`ZIP file ${zipFile.name} contains:`, Object.keys(zipContent.files));

                for (const [fileName, fileData] of Object.entries(zipContent.files)) {
                    if (!fileData.dir && fileName.match(/\.(jpg|jpeg|png|gif)$/i) && !fileName.includes('__MACOSX') && !fileName.startsWith('.')) {
                        imagePromises.push(this.processZipImage(fileName, fileData));
                    }
                }

                const images = await Promise.all(imagePromises);
                const validImages = images.filter(img => img !== null);

                if (validImages.length === 0) {
                    alert("Folder ZIP không chứa ảnh hợp lệ!");
                    return;
                }

                validImages.forEach(imageData => {
                    if (!this.uploadedImages.some(img => img.name === imageData.name)) {
                        this.uploadedImages.push(imageData);
                    }
                });
            } catch (error) {
                console.error(`Error processing ZIP file ${zipFile.name}:`, error);
                alert(`Có lỗi khi xử lý file ZIP ${zipFile.name}!`);
            }
        },

        async processZipImage(fileName, fileData) {
            try {
                const blob = await fileData.async("blob");
                const name = fileName.split('/').pop();

                const img = new Image();
                const isValid = await new Promise(resolve => {
                    img.onload = () => resolve(true);
                    img.onerror = () => resolve(false);
                    img.src = URL.createObjectURL(blob);
                });

                if (!isValid || !name.match(/\.(jpg|jpeg|png|gif)$/i)) {
                    console.warn(`Invalid or corrupted image in ZIP: ${name}`);
                    return null;
                }

                return {
                    file: new File([blob], name, { type: blob.type || "image/jpeg" }),
                    url: URL.createObjectURL(blob),
                    name: name,
                };
            } catch (error) {
                console.error(`Error processing image ${fileName}:`, error);
                return null;
            }
        },

        removeUploadedImage(index) {
            const image = this.uploadedImages[index];
            URL.revokeObjectURL(image.url);
            this.uploadedImages.splice(index, 1);
            this.predictionResult = null;
            this.currentImageIndex = -1;
        },

        clearUploadedImages() {
            if (confirm("Bạn có chắc chắn muốn xóa tất cả ảnh đã chọn?")) {
                this.uploadedImages.forEach(image => URL.revokeObjectURL(image.url));
                this.uploadedImages = [];
                this.predictionResult = null;
                this.currentImageIndex = -1;
            }
        },

        async predictImages() {
            if (this.uploadedImages.length === 0) return;

            this.isProcessing = true;
            this.predictionResult = null;

            for (let i = 0; i < this.uploadedImages.length; i++) {
                this.currentImageIndex = i;
                const image = this.uploadedImages[i];
                const formData = new FormData();
                formData.append("image", image.file);

                try {
                    const response = await PredictService.predict(formData);
                    this.predictionResult = response.data;
                    console.log(`Prediction Result for ${image.name}:`, this.predictionResult);
                    await this.saveToStore();
                } catch (error) {
                    console.error(`Error predicting image ${image.name}:`, error);
                    alert(`Có lỗi khi phân loại ảnh ${image.name}!`);
                    continue;
                }
            }

            this.isProcessing = false;
            this.currentImageIndex = -1;
            this.uploadedImages = []; // Clear after processing
            alert("Đã hoàn thành phân loại ảnh!");
        },

        async saveToStore() {
            if (!this.predictionResult || this.currentImageIndex < 0) return;

            const image = this.uploadedImages[this.currentImageIndex];
            const topClass = this.predictionResult.result.top_prediction.class_name;

            const imageDataUrl = await this.fileToDataUrl(image.file);

            const imageData = {
                name: image.name || `image_${Date.now()}.jpg`,
                url: imageDataUrl,
                confidence: this.predictionResult.result.top_prediction.percentage,
                timestamp: new Date().toISOString()
            };

            if (!this.storedImages[topClass]) {
                this.storedImages[topClass] = [];
                this.expandedFolders[topClass] = true;
            }

            this.storedImages[topClass].push(imageData);
            await this.saveToIndexedDB();
        },

        fileToDataUrl(file) {
            return new Promise((resolve, reject) => {
                const reader = new FileReader();
                reader.onload = (e) => resolve(e.target.result);
                reader.onerror = reject;
                reader.readAsDataURL(file);
            });
        },

        toggleFolder(className) {
            this.expandedFolders[className] = !this.expandedFolders[className];
            this.saveToIndexedDB();
        },

        async fetchImageAsBlob(url) {
            const response = await fetch(url);
            if (!response.ok) throw new Error(`Failed to fetch ${url}`);
            return await response.blob();
        },

        async downloadImage(image) {
            const blob = await this.fetchImageAsBlob(image.url);
            const link = document.createElement('a');
            link.href = URL.createObjectURL(blob);
            link.download = image.name;
            link.click();
            URL.revokeObjectURL(link.href);
        },

        async downloadFolder(className) {
            const zip = new JSZip();
            const folder = zip.folder(className);
            const images = this.storedImages[className];

            for (let i = 0; i < images.length; i++) {
                const image = images[i];
                const blob = await this.fetchImageAsBlob(image.url);
                folder.file(image.name || `image_${i + 1}.jpg`, blob);
            }

            zip.generateAsync({ type: "blob" }).then((content) => {
                const link = document.createElement('a');
                link.href = URL.createObjectURL(content);
                link.download = `${className}.zip`;
                link.click();
                URL.revokeObjectURL(link.href);
            });
        },

        async downloadAllImages() {
            const zip = new JSZip();
            const classNames = Object.keys(this.storedImages);

            for (const className of classNames) {
                const folder = zip.folder(className);
                const images = this.storedImages[className];

                for (let i = 0; i < images.length; i++) {
                    const image = images[i];
                    const blob = await this.fetchImageAsBlob(image.url);
                    folder.file(image.name || `image_${i + 1}.jpg`, blob);
                }
            }

            zip.generateAsync({ type: "blob" }).then((content) => {
                const link = document.createElement('a');
                link.href = URL.createObjectURL(content);
                link.download = `all_classes.zip`;
                link.click();
                URL.revokeObjectURL(link.href);
            });
        },

        deleteImage(className, imageIndex) {
            this.storedImages[className].splice(imageIndex, 1);
            if (this.storedImages[className].length === 0) {
                delete this.storedImages[className];
                delete this.expandedFolders[className];
            }
            this.saveToIndexedDB();
        },

        deleteFolder(className) {
            if (confirm(`Bạn có chắc chắn muốn xóa thư mục "${className}" và tất cả ảnh bên trong?`)) {
                delete this.storedImages[className];
                delete this.expandedFolders[className];
                this.saveToIndexedDB();
            }
        },

        handleImageError(className, imageIndex) {
            console.warn(`Image load error for ${className}[${imageIndex}]`);
        }
    },

    async mounted() {
        await this.initIndexedDB();
        await this.loadFromIndexedDB();
    },

    beforeUnmount() {
        this.uploadedImages.forEach(image => URL.revokeObjectURL(image.url));
        if (this.db) {
            this.db.close();
        }
    }
};
</script>

<style scoped>
.main {
    display: flex;
    height: 100vh;
    width: 100%;
    background-color: var(--background-main);
}

.main-container {
    display: flex;
    gap: 20px;
    width: 100%;
    padding: 20px;
    overflow: hidden;
}

/* Classify Container */
.classify-container {
    flex: 1;
    background-color: var(--background-card);
    border-radius: 15px;
    width: 100%;
    padding: 20px;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    display: flex;
    flex-direction: column;
    overflow-y: auto;
}

.section-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 20px;
    padding-bottom: 10px;
    border-bottom: 2px solid var(--border-color);
}

.section-header h3, .section-header h4 {
    margin: 0;
    color: var(--text-primary);
    display: flex;
    align-items: center;
    gap: 10px;
}

.section-header h3 {
    font-size: 1.2em;
}

.section-header h4 {
    font-size: 1.1em;
}

/* Upload Section */
.upload-mode-selector {
    display: flex;
    gap: 15px;
    margin-bottom: 15px;
    padding: 10px;
    background-color: var(--background-hover);
    border-radius: 8px;
}

.upload-mode-selector label {
    display: flex;
    align-items: center;
    gap: 5px;
    font-size: 0.9em;
    color: var(--text-primary);
    cursor: pointer;
}

.upload-mode-selector input[type="radio"] {
    accent-color: var(--primary-color);
}

.upload-loading {
    text-align: center;
    color: var(--text-secondary);
}

.upload-loading i {
    font-size: 2em;
    margin-bottom: 10px;
}

.upload-loading p {
    font-size: 1em;
    margin: 0;
}

.upload-section {
    margin-bottom: 20px;
}

.upload-area {
    border: 2px dashed var(--border-dashed);
    border-radius: 10px;
    padding: 40px;
    text-align: center;
    cursor: pointer;
    transition: all 0.3s ease;
    min-height: 200px;
    display: flex;
    align-items: center;
    justify-content: center;
}

.upload-area:hover {
    border-color: var(--primary-color);
    background-color: var(--background-hover);
}

.upload-placeholder {
    color: var(--text-secondary);
}

.upload-placeholder i {
    font-size: 3em;
    color: var(--icon-primary);
    margin-bottom: 15px;
}

.upload-placeholder p {
    font-size: 1.1em;
    margin: 10px 0 5px 0;
    font-weight: 500;
}

.upload-placeholder span {
    font-size: 0.9em;
    color: var(--text-muted);
}

.upload-loading {
    text-align: center;
    color: var(--text-secondary);
}
.upload-loading i {
    font-size: 2em;
    margin-bottom: 10px;
}
.upload-loading p {
    font-size: 1em;
    margin: 0;
}

#fileInput {
    display: none;
}

/* Uploaded Images List */
.uploaded-images-list {
    margin-bottom: 20px;
    display: flex;
    width: 100%;
    flex-direction: column;
    overflow-y: scroll;
}

.image-box {
    display: flex;
    width: 100%;
    flex-direction: column;
}

.image-box__item {
    width: 100%;
    display: flex;
    margin-bottom: 10px;
    justify-content: space-between;
}

.image-box__thumbnail {
    width: 70px;
    height: 70px;
    object-fit: cover;
    border-radius: 8px;
    margin-right: 15px;
}

.image-box__info {
    display: flex;
    flex-direction: column;
    justify-content: center;
    flex: 1;
}

.image-box__name {
    font-weight: 500;
    margin-bottom: 5px;
    color: #333;
    width: 100%;
}

.image-box__actions {
    display: flex;
    align-items: center;
    gap: 10px;
}

.image-box__actions .btn-delete-image {
    background-color: var(--accent-color);
    color: var(--border-color);
    border: none;
    border-radius: 6px;
    cursor: pointer;
    transition: background-color 0.3s ease;
    width: 30px;
    height: 30px;
}

.image-box__actions .btn-delete-image:hover {
    background-color: var(--accent-hover);
}

.image-box__actions .btn-delete-image:disabled {
    background-color: var(--disabled-color);
    cursor: not-allowed;
}

/* Action Buttons */
.action-buttons {
    display: flex;
    gap: 10px;
    margin-bottom: 20px;
}

.btn-predict, .btn-clear {
    padding: 12px 20px;
    border: none;
    border-radius: 8px;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.3s ease;
    display: flex;
    align-items: center;
    gap: 8px;
}

.btn-predict {
    background-color: var(--primary-color);;
    color: white;
    flex: 1;
}

.btn-predict:hover:not(:disabled) {
    background-color: var(--primary-hover);
}

.btn-predict:disabled {
    background-color: var(--disabled-color);
    cursor: not-allowed;
}

.btn-clear {
    background-color: var(--accent-color);
    color: white;
}

.btn-clear:hover:not(:disabled) {
    background-color: var(--accent-hover);
}

.btn-clear:disabled {
    background-color: var(--disabled-color);
    cursor: not-allowed;
}

/* Store Container */
.store-container {
    flex: 1;
    background-color: var(--background-card);
    border-radius: 15px;
    padding: 20px;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    display: flex;
    flex-direction: column;
    overflow: hidden;
}

.store-actions {
    display: flex;
    gap: 10px;
}

.btn-download-all, .btn-clear-session {
    padding: 8px 16px;
    border: none;
    border-radius: 6px;
    cursor: pointer;
    font-size: 0.9em;
    display: flex;
    align-items: center;
    gap: 6px;
}

.btn-download-all {
    background-color: var(--primary-color);
    color: white;
}

.btn-download-all:hover:not(:disabled) {
    background-color: var(--primary-hover);
}

.btn-clear-session {
    background-color: var(--accent-color);
    color: white;
}

.btn-clear-session:hover:not(:disabled) {
    background-color: var(--accent-hover);
}

.btn-download-all:disabled, .btn-clear-session:disabled {
    background-color: var(--disabled-color);
    cursor: not-allowed;
}

.folder-list {
    flex: 1;
    overflow-y: auto;
}

.folder-item {
    margin-bottom: 15px;
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    overflow: hidden;
}

.folder-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 12px 15px;
    background-color: var(--background-hover);
    cursor: pointer;
    transition: background-color 0.2s ease;
}

.folder-header:hover {
    background-color: var(--background-hover-hover);
}

.folder-info {
    display: flex;
    align-items: center;
    gap: 10px;
    font-weight: 500;
}

.folder-info i {
    color:  var(--icon-primary);
}

.image-count {
    color: var(--text-secondary);
    font-size: 0.9em;
}

.folder-actions {
    display: flex;
    gap: 5px;
}

.btn-download-folder, .btn-delete-folder {
    padding: 5px 8px;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    font-size: 0.8em;
}

.btn-download-folder {
    background-color: var(--primary-color);
    color: white;
}

.btn-download-folder:hover {
    background-color: var(--primary-hover);
}

.btn-delete-folder {
    background-color: var(--accent-color);
    color: white;
}

.btn-delete-folder:hover {
    background-color: var(--accent-hover);
}

.folder-content {
    height: 400px;
    padding: 15px;
    background-color: white;
    overflow-y: scroll;
}

.image-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
    gap: 15px;
}

.image-item {
    border: 1px solid var(--border-color);
    border-radius: 8px;
    overflow: hidden;
    background-color: var(--background-image-item);
}

.thumbnail {
    width: 100%;
    height: 80px;
    object-fit: cover;
}

.image-info {
    padding: 8px;
    font-size: 0.8em;
}

.image-name {
    display: block;
    font-weight: 500;
    margin-bottom: 4px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.image-confidence {
    color: var(--text-confidence);
    font-weight: 600;
}

.image-actions {
    display: flex;
    padding: 5px;
    gap: 5px;
}

.btn-download-image, .btn-delete-image {
    flex: 1;
    padding: 4px;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    font-size: 0.7em;
}

.btn-download-image {
    background-color: var(--primary-color);
    color: white;
}

.btn-download-image:hover {
    background-color: var(--primary-hover);
}

.btn-delete-image {
    background-color: var(--accent-color);
    color: white;
}

.btn-delete-image:hover {
    background-color: var(--accent-hover);
}

.empty-state {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    color: var(--text-secondary);
    text-align: center;
}

.empty-state i {
    font-size: 3em;
    color: var(--icon-muted);
    margin-bottom: 15px;
}

.empty-state p {
    font-size: 1.1em;
    margin: 0 0 5px 0;
}

.empty-state span {
    font-size: 0.9em;
    color: var(--text-muted);
}

/* Responsive */
@media (max-width: 768px) {
    .main-container {
        flex-direction: column;
        gap: 15px;
        padding: 15px;
    }
    
    .classify-container,
    .store-container {
        flex: none;
        min-height: 400px;
    }
    
    .image-grid {
        grid-template-columns: repeat(auto-fill, minmax(100px, 1fr));
        gap: 10px;
    }
    
    .store-actions {
        flex-wrap: wrap;
    }
}
</style>