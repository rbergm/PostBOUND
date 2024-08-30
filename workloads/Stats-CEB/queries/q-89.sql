SELECT COUNT(*)
FROM postHistory AS ph, posts AS p, users AS u
WHERE p.OwnerUserId = u.Id
  AND ph.UserId = u.Id
  AND p.Score >= -1
  AND p.CommentCount >= 0
  AND p.CommentCount <= 23
  AND u.DownVotes = 0
  AND u.UpVotes >= 0
  AND u.UpVotes <= 244;
