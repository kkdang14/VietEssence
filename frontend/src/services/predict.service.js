import createApiClient from "./api.service";

class PredictService {
    constructor(baseUrl = "/api/classify/") {
        this.api = createApiClient(baseUrl);
    }
    async predict(data) {
        const respone = await this.api.post("/", data);
        return respone;
    }
}
export default new PredictService();