import { Chart, ChartData, registerables } from "chart.js";

export class LearningCurve{
    ctx: CanvasRenderingContext2D
    data: ChartData
    chart: Chart
    constructor(ctx: CanvasRenderingContext2D, data: number[]){
        Chart.register(...registerables)
        this.ctx = ctx
        this.data = {
            labels: new Array(data.length).fill(0).map((x, index) => index),
            datasets: [{
                label: "Errors",
                data: data,
            }]
        }
        this.chart = new Chart(ctx, {
            type: 'line',
            data: this.data,
            }
        );
    }
}