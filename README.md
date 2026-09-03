# Mini Book

这个仓库同时维护多本相互独立的 MyST 笔记，并为每一本生成网页和 PDF。

| 内容 | 源目录 | 网页路径 | PDF |
| --- | --- | --- | --- |
| 计算机与深度学习 | `deep learning/` | `/computer/` | `computer-notes.pdf` |
| 金融投资 | `finance/` | `/finance/` | `finance-notes.pdf` |
| American intonation | `american intonation/` | `/american-intonation/` | `american-intonation.pdf` |
| 个人相亲简历 | `personal/` | `/resume/` | `personal-resume.pdf` |

## 本地构建

安装 Node.js、MyST 和 Typst 后，在仓库根目录执行：

```powershell
npm install -g mystmd
./scripts/build-all.ps1
```

每个项目的网页和 PDF 会生成在对应目录的 `_build/` 中。例如个人简历 PDF 位于：

```text
personal/_build/exports/personal-resume.pdf
```

PDF 构建使用仓库 `assets/fonts/` 中附带的静态 Noto Sans SC 字体，
无需在本机额外安装中文字体。

只构建一本时，进入它的目录并执行：

```powershell
myst build --typst
myst build --html
```

增量构建发生变化的书籍、启动本地服务器并打开浏览器：

```powershell
./scripts/preview.cmd
```

## 发布

推送到 `main` 分支后，GitHub Actions 会构建四本书籍，生成首页并部署到 GitHub Pages。

> `personal/cv.md` 和其中的照片会随网站公开发布。若仓库或 Pages 是公开的，请不要写入不希望公开的电话、住址、证件号码等信息。
