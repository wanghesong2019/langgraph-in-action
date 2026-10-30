import { defineConfig } from 'vitepress'
import { withMermaid } from 'vitepress-plugin-mermaid'

// VitePress 站点根 = 仓库根（.vitepress 在根目录）。
// 因此每个 `NN-xxx/README.md` 自动成为该目录的页面（URL 形如 /02-hitl/），
// 根 README.md 成为首页（/）。博客保持单一真相源，不在 docs/ 里复制。
export default withMermaid(
  defineConfig({
    lang: 'zh-CN',
    title: 'LangGraph 实战入门',
    description: '每个知识点 = 可运行 demo + 技术博客 + 标注到章节的参考文献',

    // 项目页站点挂在 /<repo>/ 下（非用户根域名），必须显式设 base，否则资源与链接全 404
    base: '/langgraph-in-action/',

    // 排除本地内部文档与下载的官方原文（已被 gitignore，但本地构建时仍存在，防泄漏）
    srcExclude: ['写作规范.md', '**/Interrupts.md'],

    // VitePress 不像 VuePress 那样默认把 README.md 当目录索引，需显式 rewrite：
    // 根 README.md → 首页 /；NN-xxx/README.md → /NN-xxx/（GitHub Pages 可直接按目录解析）
    rewrites: {
      'README.md': 'index.md',
      ':dir/README.md': ':dir/index.md',
    },

    themeConfig: {
      nav: [
        { text: 'GitHub 仓库', link: 'https://github.com/wanghesong2019/langgraph-in-action' },
      ],
      sidebar: [
        {
          text: '系列目录',
          items: [
            { text: '系列导航', link: '/' },
            { text: '第 1 讲 · LangGraph 入门', link: '/01-state-graph/' },
            { text: '第 2 讲 · HITL 人工审批', link: '/02-hitl/' },
            { text: '第 3 讲 · 多 Agent Supervisor', link: '/03-supervisor-multiagent/' },
          ],
        },
      ],
      outline: { level: [2, 3] },
      docFooter: { prev: '上一篇', next: '下一篇' },
      lastUpdated: true,
    },
  }),
)
