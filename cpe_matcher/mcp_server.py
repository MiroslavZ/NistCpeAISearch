"""Local stdio MCP server; stdout belongs exclusively to MCP."""
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from .nvd import NvdService, NvdServiceError


def _invoke(method, *args):
    # MCP 2 masks unexpected exceptions. Explicit ToolError keeps the safe
    # infrastructure marker visible so the CLI stops the report on outages.
    try:
        return method(*args)
    except (NvdServiceError, ValueError) as exc:
        raise ToolError(str(exc)) from exc


def create_server(service: NvdService) -> MCPServer:
    server = MCPServer('NVD CPE Matcher')

    @server.tool()
    def search_cpe(keyword: str = '', part: str = '', vendor: str = '', product: str = '', limit: int = 50) -> dict:
        """Найти группы продуктов CPE по гипотезе компонентов или ключевым словам.

        Перед поиском по vendor/product преобразуй сырые названия в гипотезу CPE:
        восстанови оригинальное название вместо кириллицы, обычно используй
        строчные ASCII-символы и подчёркивания вместо пробелов. Экранируй специальные
        символы согласно CPE и JSON. Например, «Корпорация Майкрософт» -> microsoft.
        Гипотеза не является подтверждённым соответствием; оцени возвращённые записи.
        part: a (приложение), o (ОС), h (оборудование) или пусто, если неизвестно.
        vendor/product: гипотезы компонентов CPE или точные компоненты из NVD;
        неизвестные поля оставляй пустыми. Не передавай полную CPE в одно поле.
        keyword: исходное или преобразованное название; разрешены Unicode и пробелы.
        При keyword-поиске необязательно задавать компоненты: фильтры применяются вместе.
        incomplete означает неполную выдачу; evidence содержит deprecated и замены.
        """
        return _invoke(service.search_cpe, keyword, part, vendor, product, limit)

    @server.tool()
    def search_cve(keyword: str, limit: int = 10) -> dict:
        """Find CVE context and CPE hypotheses. Preserve vulnerable flags and configuration structure; never proves installed vulnerabilities."""
        return _invoke(service.search_cve, keyword, limit)

    @server.tool()
    def validate_product(part: str, vendor: str, product: str) -> dict:
        """Verify exact CPE triple across all versions. Only valid confirms an active dictionary entry; inspect replacements if deprecated_only."""
        return _invoke(service.validate_product, part, vendor, product)

    @server.tool()
    def validate_vendor(vendor: str) -> dict:
        """Verify an exact vendor component using an active CPE dictionary record."""
        return _invoke(service.validate_vendor, vendor)

    return server


def main():
    import os
    from dotenv import load_dotenv
    from .logging_setup import configure_logging
    load_dotenv()
    configure_logging(os.getenv('CPE_MATCH_VERBOSE') == '1')
    create_server(NvdService.from_env()).run(transport='stdio')


if __name__ == '__main__':
    main()
