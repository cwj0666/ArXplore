import type { PaperListItem } from "../pages/list/listTypes";


export interface BootstrapPayload {
  is_authenticated: boolean;
  username: string;
  has_personal_api_key: boolean;
  preferred_summary_model: string;
  available_summary_models: string[];
  demo_mode?: boolean;
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
