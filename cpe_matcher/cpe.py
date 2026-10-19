"""Small CPE 2.3 formatted-string helpers; retain dictionary spelling."""


def split_cpe(name: str) -> tuple[str, ...]:
    fields, current = [], []
    escaped = False
    for char in name:
        if char == ':' and not escaped:
            fields.append(''.join(current))
            current = []
        else:
            current.append(char)
        escaped = char == '\\' and not escaped
    fields.append(''.join(current))
    if escaped or len(fields) != 13 or fields[:2] != ['cpe', '2.3']:
        raise ValueError('Некорректная строка CPE 2.3')
    return tuple(fields[2:])


def component(value: str) -> str:
    """Accept a concrete bound component, rejecting unescaped pattern syntax."""
    if not value or value in ('*', '-') or len(value) > 512:
        raise ValueError('Нужен конкретный компонент CPE')
    escaped = False
    for char in value:
        if char.isspace() or ord(char) < 32 or not char.isascii():
            raise ValueError(
                'Компонент CPE должен использовать ASCII без пробелов и управляющих символов. '
                'Преобразуйте сырое название в гипотезу оригинального имени CPE '
                '(обычно слова разделяются подчёркиванием) или используйте keyword-поиск.')
        if not escaped and char != '\\' and not (char.isalnum() or char in '._-'):
            raise ValueError(
                'Компонент CPE содержит неэкранированные спецсимволы. '
                r'Используйте компоненты из словаря; например visual_c\+\+ вместо visual_c++.')
        escaped = char == '\\' and not escaped
    if escaped:
        raise ValueError('Незавершённое экранирование CPE')
    return value


def match_string(part: str = '', vendor: str = '', product: str = '') -> str:
    if part and part not in ('a', 'o', 'h'):
        raise ValueError('part должен быть a, o или h')
    return 'cpe:2.3:' + ':'.join([
        part or '*', component(vendor) if vendor else '*',
        component(product) if product else '*', *(['*'] * 8),
    ])
