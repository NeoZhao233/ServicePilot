"""Create small synthetic after-sales PDFs for the repeatable PDF evaluation."""

from pathlib import Path

from PIL import Image, ImageDraw
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import Table, TableStyle

ROOT = Path(__file__).parent / "pdf_fixtures"
ROOT.mkdir(exist_ok=True)
pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))


def page(canvas, title):
    canvas.setFont("STSong-Light", 17)
    canvas.drawString(48, 780, title)
    canvas.setFont("STSong-Light", 11)


def make_policy():
    canvas = Canvas(str(ROOT / "policy.pdf"), pagesize=A4)
    page(canvas, "售后政策")
    canvas.drawString(48, 730, "签收后七日内，未拆封商品可申请无理由退货。")
    canvas.drawString(48, 700, "质量故障不受未拆封条件限制，需提供检测材料。")
    canvas.drawString(48, 670, "退款将在审核通过后三个工作日内原路退回。")
    canvas.showPage()
    canvas.save()


def make_table():
    canvas = Canvas(str(ROOT / "conditions.pdf"), pagesize=A4)
    page(canvas, "退货条件表")
    rows = [
        ["商品状态", "申请期限", "处理方式"],
        ["未拆封", "签收后7日", "无理由退货"],
        ["质量故障", "保修期内", "检测后维修或换货"],
        ["人为损坏", "不适用", "付费维修"],
    ]
    table = Table(rows, colWidths=[125, 155, 220], rowHeights=40)
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), "STSong-Light"),
                ("FONTSIZE", (0, 0), (-1, -1), 11),
                ("GRID", (0, 0), (-1, -1), 1, colors.black),
            ]
        )
    )
    table.wrapOn(canvas, 500, 300)
    table.drawOn(canvas, 48, 480)
    canvas.showPage()
    canvas.save()


def board_image():
    image = Image.new("RGB", (420, 260), "#dddddd")
    draw = ImageDraw.Draw(image)
    draw.rectangle((30, 25, 390, 235), fill="#277152", outline="black", width=5)
    for x in (75, 155, 235, 315):
        draw.rectangle((x, 65, x + 35, 100), fill="#333333")
        draw.line((x, 145, x + 70, 145), fill="#d5b46b", width=6)
    draw.ellipse((245, 110, 305, 165), fill="#222222")
    return image


def make_report():
    canvas = Canvas(str(ROOT / "inspection.pdf"), pagesize=A4)
    for number in (1, 2):
        page(canvas, f"检测报告 第{number}页")
        canvas.drawString(48, 730, "设备编号 SP-001，已进入检测流程。")
        canvas.showPage()
    page(canvas, "检测报告 第3页")
    canvas.drawString(48, 730, "文字检测结论：设备主板损坏，建议更换主板。")
    canvas.drawString(48, 690, "图1：主板检测照片。")
    canvas.drawImage(ImageReader(board_image()), 80, 400, width=420, height=260)
    canvas.showPage()
    canvas.save()


def make_scan():
    image = Image.new("RGB", (900, 1200), "white")
    draw = ImageDraw.Draw(image)
    draw.text((70, 100), "REPAIR RECEIPT", fill="black", font_size=44)
    draw.text((70, 220), "Device SP-200", fill="black", font_size=32)
    draw.text((70, 300), "Repair date: 2026-09-20", fill="black", font_size=32)
    draw.text((70, 380), "Warranty: 12 months", fill="black", font_size=32)
    canvas = Canvas(str(ROOT / "scan_receipt.pdf"), pagesize=A4)
    canvas.drawImage(ImageReader(image), 0, 0, width=A4[0], height=A4[1])
    canvas.showPage()
    canvas.save()


def make_decorated():
    canvas = Canvas(str(ROOT / "decorated.pdf"), pagesize=A4)
    page(canvas, "售后说明")
    canvas.drawString(48, 730, "保修凭证须包含商品序列号与购买日期。")
    logo = Image.new("RGB", (32, 32), "#578fd0")
    canvas.drawImage(ImageReader(logo), 50, 50, width=32, height=32)
    canvas.showPage()
    canvas.save()


def make_invoice():
    canvas = Canvas(str(ROOT / "invoice.pdf"), pagesize=A4)
    page(canvas, "配件采购发票")
    canvas.drawString(48, 745, "发票编号 INV-2026-123；购买日期 2026-09-21。")
    rows = [["配件编号", "数量", "单价", "小计"]]
    for number in range(1, 13):
        rows.append([f"A{number:02d}", "1", f"{number * 10}元", f"{number * 10}元"])
    rows.append(["合计", "12", "", "780元"])
    table = Table(rows, colWidths=[130, 70, 130, 170], rowHeights=31)
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), "STSong-Light"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("GRID", (0, 0), (-1, -1), 0.7, colors.black),
            ]
        )
    )
    table.wrapOn(canvas, 500, 550)
    table.drawOn(canvas, 48, 200)
    canvas.showPage()
    canvas.save()


def make_manual():
    canvas = Canvas(str(ROOT / "manual.pdf"), pagesize=A4)
    page(canvas, "家用路由器说明书")
    canvas.drawString(48, 720, "型号 R-500。安装前请断开电源。")
    canvas.showPage()
    page(canvas, "重置与故障排查")
    canvas.drawString(48, 730, "设备无法联网时，按住重置按钮十秒，等待指示灯闪烁。")
    canvas.drawString(48, 690, "重置后重新输入无线网络密码。")
    canvas.drawString(48, 650, "图2：路由器背面接口示意。")
    drawing = Image.new("RGB", (420, 210), "#e6e6e6")
    draw = ImageDraw.Draw(drawing)
    draw.rounded_rectangle((30, 40, 390, 170), radius=20, fill="#444444")
    draw.ellipse((310, 85, 335, 110), fill="red")
    canvas.drawImage(ImageReader(drawing), 80, 350, width=420, height=210)
    canvas.showPage()
    canvas.save()


def make_gallery():
    canvas = Canvas(str(ROOT / "gallery.pdf"), pagesize=A4)
    page(canvas, "产品部件图册")
    canvas.drawString(48, 730, "图A：左侧充电接口。")
    canvas.drawString(310, 730, "图B：右侧电池仓。")
    left = Image.new("RGB", (230, 180), "#afcef0")
    right = Image.new("RGB", (230, 180), "#e6d1a7")
    ImageDraw.Draw(left).rectangle((70, 65, 160, 120), fill="#333333")
    ImageDraw.Draw(right).rectangle((50, 40, 180, 140), fill="#666666")
    canvas.drawImage(ImageReader(left), 48, 510, width=230, height=180)
    canvas.drawImage(ImageReader(right), 310, 510, width=230, height=180)
    canvas.showPage()
    canvas.save()


if __name__ == "__main__":
    make_policy()
    make_table()
    make_report()
    make_scan()
    make_decorated()
    make_invoice()
    make_manual()
    make_gallery()
