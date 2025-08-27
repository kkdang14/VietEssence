<template>
    <div class="welcome-container" :style="containerStyle">
        <div class="background"></div>
        <div class="content">
            <h1 class="fade-in">Welcome to VietEssence</h1>
            <p class="fade-in delay">Application for classification of Intangible Cultures</p>
            <router-link to="/predict">
                <button class="welcome-btn fade-in delay-more">Classify now</button>
            </router-link>
        </div>
    </div>
</template>

<script>
export default {
    name: "WelcomeView",
    methods: {
        showAlert() {
            alert("Chúc bạn có trải nghiệm tuyệt vời với Vue.js!");
        },
    },
};
</script>

<style scoped>
.welcome-container {
    position: relative;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 100vh;
    text-align: center;
    background: linear-gradient(135deg, #f9f5ed, #e6dfd5);
    color: var(--primary-color);
    font-family: 'Noticia Text', Arial, sans-serif;
    overflow: hidden;
}

.background {
    position: absolute;
    top: 50%;
    left: 50%;
    width: 750px;
    height: 750px;
    background-image: url('@/assets/trong-dong.png');
    background-size: contain;
    background-repeat: no-repeat;
    background-position: center;
    transform: translate(-50%, -50%);
    animation: spin 22s linear infinite;
    opacity: 0.5;
    z-index: 1;
    
    /* Single powerful sunshine glow */
    filter: drop-shadow(0 0 50px rgba(255, 215, 0, 0.9));
}

/* Single sunshine glow layer */
.background::before {
    content: '';
    position: absolute;
    top: -80px;
    left: -80px;
    right: -80px;
    bottom: -80px;
    background: radial-gradient(
        circle, 
        rgba(255, 215, 0, 0.6) 0%, 
        rgba(255, 180, 0, 0.4) 40%, 
        transparent 70%
    );
    border-radius: 50%;
    animation: sunshineGlow 3s ease-in-out infinite alternate;
    z-index: -1;
}

.content {
    position: relative;
    z-index: 2;
}

h1 {
    font-size: 2.5rem;
    margin-bottom: 20px;
    max-width: 800px;
    line-height: 1.3;
}

p {
    font-size: 1.2rem;
    margin-bottom: 30px;
    max-width: 700px;
    line-height: 1.5;
}

.welcome-btn {
    padding: 12px 25px;
    font-size: 1.1rem;
    color: #fff;
    background: var(--primary-color);
    border: none;
    border-radius: 20px;
    cursor: pointer;
    transition: transform 0.2s, background 0.3s;
    box-shadow: 0 3px 6px rgba(107, 91, 78, 0.3);
}

.welcome-btn:hover {
    background: #5a422c;
    transform: scale(1.05);
}

/* Fade-in animations */
.fade-in {
    opacity: 0;
    animation: fadeIn 1s ease-in forwards;
}

.delay {
    animation-delay: 0.5s;
}

.delay-more {
    animation-delay: 1s;
}

/* Spin animation for the bronze drum */
@keyframes spin {
    0% { 
        transform: translate(-50%, -50%) rotate(0deg); 
    }
    100% { 
        transform: translate(-50%, -50%) rotate(360deg); 
    }
}

/* Powerful sunshine glow animation */
@keyframes sunshineGlow {
    0% { 
        opacity: 0.4;
        transform: scale(1);
    }
    100% { 
        opacity: 0.8;
        transform: scale(1.2);
    }
}

@keyframes fadeIn {
    from {
        opacity: 0;
        transform: translateY(10px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}
</style>