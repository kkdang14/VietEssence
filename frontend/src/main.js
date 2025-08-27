import { createApp } from 'vue'
import App from './App.vue'
import router from "./router"
import "@/assets/root_color.css"
import "bootstrap/dist/css/bootstrap.min.css"
import "@fortawesome/fontawesome-free/css/all.min.css"

createApp(App).use(router).mount('#app')