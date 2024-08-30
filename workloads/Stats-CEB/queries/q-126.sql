SELECT COUNT(*)
FROM postHistory AS ph,
  votes AS v,
  users AS u,
  badges AS b
WHERE u.Id = b.UserId
  AND u.Id = ph.UserId
  AND u.Id = v.UserId
  AND u.Views >= 0;
