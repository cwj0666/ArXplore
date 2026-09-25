import type { PaperListItem } from "../pages/list/listTypes";


export interface BootstrapPayload {
  is_authenticated: boolean;
  username: string;
  has_personal_api_key: boolean;
  preferred_summary_model: string;
  available_summary_models: string[];
  /** 로그인 없이 목록·상세와 캐시된 AI 결과를 볼 수 있는지 여부 */
  demo_mode?: boolean;
  /** 로그인이 필요한 기능 식별자 목록 (서버가 정한다) */
  login_required_for?: string[];
}


export interface AuthPayload {
  ok?: boolean;
  username?: string;
}


export interface SettingsPayload {
  preferred_summary_model: string;
  available_summary_models: string[];
}


export interface FavoriteListPayload {
  items: PaperListItem[];
}


export interface FavoriteTogglePayload {
  is_favorited?: boolean;
}


export interface ApiKeyPayload {
  ok?: boolean;
  has_personal_api_key?: boolean;
}
