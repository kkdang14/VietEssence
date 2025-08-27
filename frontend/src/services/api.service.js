import axios from "axios";

const commonConfig = {
    headers: {
        "Content-Type": "multipart/form-data", 
        Accept: "application/json",
    },
};
export default (baseURL) => {
    return axios.create({
        baseURL,
        ...commonConfig,
    });
};
