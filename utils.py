def get_pet_evolution_data(pet_type: str, level: int, current_xp: int) -> dict:
    evolution_lines = {
        'nova': ['🥚', '🦊', '🔥', '🐉'],
        'pip':  ['🥚', '🐧', '🦅', '🦅'],
        'luna': ['🧶', '🐱', '🐈‍⬛', '🐈‍⬛'],
        'zap':  ['🔋', '🐶', '⚡', '⚡']
    }
    
    line = evolution_lines.get(pet_type.lower(), ['🥚', '✨', '🌟', '💫'])
    
    if level <= 3:
        stage, next_lvl = 0, 4
    elif level <= 7:
        stage, next_lvl = 1, 8
    elif level <= 12:
        stage, next_lvl = 2, 13
    else:
        stage, next_lvl = 3, level

    target_xp = level * 100 
    is_maxed = stage == 3
    progress_percent = 1.0 if is_maxed else min(current_xp / target_xp, 1.0)
    
    return {
        "current_emoji": line[stage],
        "next_emoji": line[stage] if is_maxed else line[stage + 1],
        "progress_percent": progress_percent,
        "is_maxed": is_maxed,
        "xp_to_next": target_xp
    }