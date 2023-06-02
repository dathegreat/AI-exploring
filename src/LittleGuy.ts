export const drawLittleGuy = (ctx: CanvasRenderingContext2D, center: {x: number, y: number}, pixelSize: number) =>{
    ctx.translate(center.x, center.y)
    const legTheta = Math.PI / 3
    const bodyRadius = pixelSize * 2
    const leftLegStart = {
      x: -(Math.cos(legTheta) * bodyRadius), 
      y: (Math.sin(legTheta) * bodyRadius)
    }
    const rightLegStart = {
      x: (Math.cos(legTheta) * bodyRadius), 
      y: (Math.sin(legTheta) * bodyRadius)
    }
    //draw body
    ctx.beginPath()
    ctx.arc(0, 0, pixelSize * 2, 0, Math.PI * 2)
    ctx.stroke()
    ctx.closePath()
    //draw legs
    ctx.beginPath()
    ctx.moveTo(leftLegStart.x, leftLegStart.y)
    ctx.lineTo(leftLegStart.x * 1.5, leftLegStart.y * 1.5)
    ctx.moveTo(rightLegStart.x, rightLegStart.y)
    ctx.lineTo(rightLegStart.x * 1.5, rightLegStart.y * 1.5)
    ctx.stroke()
    ctx.closePath()
    //draw eyes
    ctx.beginPath()
    ctx.arc(pixelSize, -pixelSize / 2, pixelSize / 4, 0, Math.PI * 2)
    ctx.stroke()
    ctx.beginPath()
    ctx.arc(-pixelSize, -pixelSize / 2, pixelSize / 4, 0, Math.PI * 2)
    ctx.stroke()
    //draw mouth
    ctx.beginPath()
    ctx.moveTo(-pixelSize / 2, pixelSize / 4)
    ctx.bezierCurveTo(-pixelSize, pixelSize, pixelSize, pixelSize, pixelSize / 2, pixelSize / 4)
    ctx.stroke()
    //draw hat
    ctx.translate(0, -bodyRadius)
    ctx.moveTo(0,0)
    ctx.lineTo(-bodyRadius, 0)
    ctx.lineTo(-bodyRadius, -pixelSize)
    ctx.lineTo(-bodyRadius / 2, -pixelSize)
    ctx.lineTo(-bodyRadius / 2, -pixelSize * 2.5)
    ctx.lineTo(bodyRadius / 2, -pixelSize * 2.5)
    ctx.lineTo(bodyRadius / 2, -pixelSize)
    ctx.lineTo(bodyRadius, -pixelSize)
    ctx.lineTo(bodyRadius, 0)
    ctx.lineTo(0,0)
    ctx.stroke()
  }