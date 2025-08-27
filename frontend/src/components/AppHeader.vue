<template>
    <nav class="navbar">
        <div class="navbar-brand-container">
            <a href="/" class="navbar-brand">
                <!-- <span class="brand-icon"><img src="../assets/logo2.png" alt=""></span> -->
                <!-- <span class="brand-text">Viet Instruments</span> -->
            </a>
            <div class="navbar-nav">
                <!-- <li class="nav-item"><b>Home</b></li>
                <li class="nav-item"><b>Contact</b></li>     -->
            </div>
        </div>
    </nav>
</template>

<script>
export default {
    name: "app-header",
};
</script>

<style scoped>
.logo {
    width: 50px;
    height: auto;
}

.brand-icon {
    width: 120px;
    height: 120px;
    margin-bottom: 10px;

}

.brand-icon img {
    width: 100%; /* hoặc auto, hoặc 100% tùy theo mục đích */
    height:100%;
    object-fit: cover;
    border-radius: 50%;
    border:solid 2px #58493d;
}

.brand-text {
    font-size: 18px;
    font-weight: bold;
    color: #6b5b4e;
    white-space: nowrap;
}

.navbar {
    background-color: #f0e9df;
    height: 100vh;
    width: 250px;
    box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
    display: flex;
    border-right: 1px solid #e0d6c7;
}

.navbar-brand-container{
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    width: 100%;
    height: 100%;
    padding: 20px 0;
}

.navbar-brand{
    display: flex;
    justify-content: center;
    align-items: center;
    flex-direction: column;
    text-decoration: none;
    padding: 15px 0;
    transition: transform 0.2s ease;
}

.navbar-brand:hover {
    transform: scale(1.05);
}

.navbar-nav{
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    padding: 0 20px;
}

.navbar-nav .nav-item{
    color: #6b5b4e;
    padding: 10px 0;
    cursor: pointer;
    transition: color 0.2s ease;
    list-style-type: none;
}

.navbar-nav .nav-item:hover {
    color: #857464;
}
</style>